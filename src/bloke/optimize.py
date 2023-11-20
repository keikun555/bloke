"""Bril optimizer using STOKE"""

import copy
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Iterable, TypeAlias, cast

import click
import numpy as np
import numpy.typing as npt

from bloke.bril_equivalence import (
    EquivalenceAnalysisResult,
    z3_prove_equivalence_or_find_counterexample,
)
from bloke.mcmc import MonteCarloMarkovChainSample, Probability
from bril.bril2z3 import COMPATIBLE_OPS
from bril.bril_constants import OPERATORS, GenericType
from bril.brili import Brili, Brilirs, SubprocessBrili
from bril.briltxt import prints_prog
from bril.typing_bril import (
    BrilType,
    Effect,
    Instruction,
    Operation,
    PrimitiveType,
    Program,
    Value,
    Variable,
)

Transformation: TypeAlias = Callable[[Program], Program]
TransformationGenerator: TypeAlias = Callable[[Program], Transformation | None]

TransformationUndo: TypeAlias = Callable[[Program], Program]

logger = logging.getLogger(__name__)

MINIMUM_NUM_ARGS = 3


class Cost(float):
    ...


def transformation_eval(transformation: Transformation, program: Program) -> Program:
    return transformation(program)


def operations_dicts_get() -> tuple[
    dict[
        tuple[tuple[BrilType | GenericType, ...], BrilType | GenericType | None],
        list[Operation],
    ],
    dict[
        Operation,
        tuple[tuple[BrilType | GenericType, ...], BrilType | GenericType | None],
    ],
]:
    arg_types_to_operations_dict: dict[
        tuple[tuple[BrilType | GenericType, ...], BrilType | GenericType | None],
        list[Operation],
    ] = defaultdict(list)
    operation_dict: dict[
        Operation,
        tuple[tuple[BrilType | GenericType, ...], BrilType | GenericType | None],
    ] = {}

    for operation, argument_types, return_type in OPERATORS:
        if operation not in COMPATIBLE_OPS or operation in ("phi", "const"):
            continue
        arg_types_to_operations_dict[(argument_types, return_type)].append(operation)
        operation_dict[operation] = (argument_types, return_type)

    return dict(arg_types_to_operations_dict), operation_dict


TestCase: TypeAlias = tuple[tuple[PrimitiveType, ...], int]


def fresh_variable(variables: Iterable[Variable]):
    variable_set = set(variables)
    prefix = "x"
    i = 0
    while True:
        candidate = f"{prefix}{i}"
        if candidate not in variable_set:
            return candidate

        i += 1


def one_testcase_validation(
    brili: Brili, program: Program, testcase: TestCase
) -> float:
    """Calculate validation score for one test case"""
    arguments, expected_output = testcase
    result = brili.interpret(program, arguments, profile=False)
    if result.error is not None:
        # In the future we may want to give different kinds of scores for different errors
        logger.debug(result.error)
        return 10.0

    return 10.0 * float(abs(result.returncode - expected_output))


def calculate_validation(
    brili: Brili, program: Program, test_cases: set[TestCase]
) -> float:
    """Calculate validation score for test cases"""
    if len(test_cases) <= 0:
        return 0.0

    validator = partial(one_testcase_validation, brili, program)
    return sum(map(validator, test_cases))


def calculate_performance(program: Program) -> float:
    """Calculate expected performance"""
    # Number of instructions
    return float(
        sum(
            len(
                list(
                    filter(
                        lambda i: "op" in i and cast(Effect, i)["op"] != "nop",
                        func["instrs"],
                    )
                )
            )
            for func in program["functions"]
        )
    )


def random_instruction_in_program(
    program: Program, qualifier: Callable[[Instruction], bool] | None = None
) -> tuple[str, int, Instruction] | None:
    """Return random instruction in a program
    Returns an (function name, instruction index, instruction) tuple"""
    instructions: list[tuple[str, int, Instruction]] = []
    for function in program["functions"]:
        for i, instruction in enumerate(function["instrs"]):
            if qualifier is not None and not qualifier(instruction):
                continue
            instructions.append((function["name"], i, instruction))

    if len(instructions) <= 0:
        return None

    return instructions[np.random.choice(len(instructions))]


class BlokeSample(MonteCarloMarkovChainSample[Program]):
    """Does MCMC on Bril programs"""

    def __init__(
        self,
        brili: Brili,
        initial_program: Program,
        opcode_weight: int,
        operand_weight: int,
        swap_weight: int,
        instruction_weight: int,
        unused_probability: Probability | None,
        mcmc_beta: float,
    ):
        super().__init__(mcmc_beta)

        self._initial_program = initial_program

        self._verification_cache: dict[str, EquivalenceAnalysisResult] = {}

        # Interpreter
        self.brili = brili

        # For opcode transformation generation
        (
            self._arg_types_to_operations_dict,
            self._operation_dict,
        ) = operations_dicts_get()

        # For finding the next candidate
        self._transform_weights = np.array(
            [opcode_weight, operand_weight, swap_weight, instruction_weight]
        )
        self._normalized_weights: npt.NDArray[np.float64] = np.array(
            self._transform_weights
        ) / np.sum(self._transform_weights)

        self._unused_probability: Probability | None = unused_probability

        # For the cost function
        self.test_cases: set[TestCase] = set()
        self.performance_correctness_ratio: float = 0.0

        self._function_to_variables: dict[
            str, tuple[dict[Variable, BrilType], dict[BrilType, list[Variable]]]
        ] = {}
        for function in initial_program["functions"]:
            function_name = function["name"]
            self._function_to_variables[function_name] = (
                {},
                {"int": [], "float": [], "bool": []},
            )
            variable_to_type, type_to_variables = self._function_to_variables[
                function_name
            ]
            for arg in function["args"]:
                variable_to_type[arg["name"]] = arg["type"]
                type_to_variables[arg["type"]].append(arg["name"])
            for instruction in function["instrs"]:
                if "dest" in instruction:
                    value = cast(Value, instruction)
                    variable_to_type[value["dest"]] = value["type"]
                    type_to_variables[value["type"]].append(value["dest"])

            for type_str in ("int", "float", "bool"):
                type_ = cast(BrilType, type_str)

                num_args = len(type_to_variables[type_])
                if num_args < MINIMUM_NUM_ARGS:
                    for _ in range(MINIMUM_NUM_ARGS - num_args):
                        var = fresh_variable(variable_to_type.keys())
                        variable_to_type[var] = type_
                        type_to_variables[type_].append(var)

    def _variable_type_dicts_get(
        self, program: Program, function_name: str
    ) -> tuple[dict[Variable, BrilType], dict[BrilType, list[Variable]]]:
        variable_to_type: dict[Variable, BrilType] = {}
        type_to_variables: dict[BrilType, list[Variable]] = defaultdict(list)
        for function in program["functions"]:
            if function["name"] != function_name:
                continue

            if "args" in function:
                # Function arguments are hidden variables
                for arg in function["args"]:
                    variable_to_type[arg["name"]] = arg["type"]
                    type_to_variables[arg["type"]].append(arg["name"])

            for instruction in function["instrs"]:
                # Actual variables defined in Value instructions
                if "dest" in instruction:
                    value = cast(Value, instruction)
                    variable_to_type[value["dest"]] = value["type"]
                    type_to_variables[value["type"]].append(value["dest"])

            break

        return variable_to_type, type_to_variables

    def _random_opcode_transformation(self, program: Program) -> Transformation | None:
        random_instruction = random_instruction_in_program(
            program, lambda i: "op" in i and "value" not in i
        )
        if random_instruction is None:
            return None
        (
            function_name,
            instruction_index,
            instruction,
        ) = random_instruction

        operation = cast(Effect, instruction)["op"]

        opcode_argument_types, opcode_return_type = self._operation_dict[operation]
        operands_with_same_arity_and_types = self._arg_types_to_operations_dict[
            (opcode_argument_types, opcode_return_type)
        ]
        random_operand = np.random.choice(operands_with_same_arity_and_types)

        def transformation(program: Program) -> Program:
            new_program: Program = copy.deepcopy(program)

            for function in new_program["functions"]:
                if function["name"] != function_name:
                    continue
                effect = cast(Effect, function["instrs"][instruction_index])
                effect["op"] = random_operand

                break

            return new_program

        return transformation

    def _random_operand_transformation(self, program: Program) -> Transformation | None:
        random_instruction = random_instruction_in_program(
            program, lambda i: "args" in i and len(cast(Effect, i)["args"]) > 0
        )
        if random_instruction is None:
            return None
        (
            function_name,
            instruction_index,
            instruction,
        ) = random_instruction

        _, type_to_variables = self._function_to_variables[function_name]

        effect = cast(Effect, instruction)

        operand_index = np.random.choice(len(effect["args"]))
        operand_type: BrilType
        bril_type_or_generic = self._operation_dict[effect["op"]][0][operand_index]
        if bril_type_or_generic == "generic":
            if "dest" in effect:
                value = cast(Value, effect)
                operand_type = value["type"]
            else:
                # TODO do something else here if we want to support "call" and "print"
                operand_type = "int"

        else:
            operand_type = cast(BrilType, bril_type_or_generic)
        random_variable = np.random.choice(type_to_variables[operand_type])

        def transformation(program: Program) -> Program:
            new_program: Program = copy.deepcopy(program)

            for function in new_program["functions"]:
                if function["name"] != function_name:
                    continue
                effect = cast(Effect, function["instrs"][instruction_index])
                effect["args"][operand_index] = random_variable

                break

            return new_program

        return transformation

    def _random_swap_transformation(self, program: Program) -> Transformation | None:
        random_instruction1 = random_instruction_in_program(program)
        random_instruction2 = random_instruction_in_program(program)
        if random_instruction1 is None or random_instruction2 is None:
            return None
        (
            function1_name,
            instruction1_index,
            instruction1,
        ) = random_instruction1
        (
            function2_name,
            instruction2_index,
            instruction2,
        ) = random_instruction2

        def transformation(program: Program) -> Program:
            new_program: Program = copy.deepcopy(program)

            swap1, swap2 = False, False
            for function in new_program["functions"]:
                if function["name"] == function1_name:
                    function["instrs"][instruction1_index] = instruction2
                    swap1 = True
                if function["name"] == function2_name:
                    function["instrs"][instruction2_index] = instruction1
                    swap2 = True

                if swap1 and swap2:
                    break

            return new_program

        return transformation

    def _random_instruction_transformation(
        self, program: Program
    ) -> Transformation | None:
        random_instruction_to_replace = random_instruction_in_program(
            program, lambda i: "label" not in i
        )
        if random_instruction_to_replace is None:
            return None
        (
            function_name,
            instruction_index,
            _,
        ) = random_instruction_to_replace

        variable_to_type, type_to_variables = self._function_to_variables[function_name]

        random_instruction: Effect | Value

        if (
            self._unused_probability is not None
            and np.random.rand() < self._unused_probability
        ):
            random_instruction = {"op": "nop"}
        else:

            def filter_(operation: Operation) -> bool:
                if operation in ("phi", "const"):
                    return False

                argument_types, _ = self._operation_dict[operation]
                for type_ in argument_types:
                    if (
                        type_ != "generic"
                        and len(type_to_variables[cast(BrilType, type_)]) <= 0
                    ):
                        return False
                    if type == "generic" and len(variable_to_type) <= 0:
                        return False

                return True

            random_operation = np.random.choice(
                list(
                    filter(
                        filter_,
                        COMPATIBLE_OPS,
                    )
                )
            )
            random_instruction = {"op": random_operation}

            argument_types, return_type = self._operation_dict[random_operation]
            arguments: list[Variable] = []
            for type_ in argument_types:
                if type_ == "generic":
                    if len(variable_to_type) <= 0:
                        return None
                    generic_type: BrilType = variable_to_type[
                        np.random.choice(list(variable_to_type.keys()))
                    ]
                    variables = type_to_variables[generic_type]
                else:
                    variables = type_to_variables[cast(BrilType, type_)]
                random_operand = np.random.choice(variables)
                arguments.append(random_operand)

            if len(arguments) > 0:
                random_instruction["args"] = arguments

            if return_type is not None:
                if return_type == "generic":
                    if len(variable_to_type) <= 0:
                        return None
                    generic_type = variable_to_type[
                        np.random.choice(list(variable_to_type.keys()))
                    ]
                    return_type = generic_type
                    variables = type_to_variables[generic_type]
                else:
                    variables = type_to_variables[cast(BrilType, return_type)]

                # variables.append(fresh_variable(variable_to_type.keys()))

                cast(Value, random_instruction)["type"] = cast(BrilType, return_type)
                cast(Value, random_instruction)["dest"] = np.random.choice(variables)

        def transformation(program: Program) -> Program:
            new_program: Program = copy.deepcopy(program)

            for function in new_program["functions"]:
                if function["name"] != function_name:
                    continue

                function["instrs"][instruction_index] = random_instruction
                break

            return new_program

        return transformation

    def _next_candidate(
        self, program: Program
    ) -> tuple[Program, Probability, Probability]:
        generators: tuple[TransformationGenerator, ...] = (
            self._random_opcode_transformation,
            self._random_operand_transformation,
            self._random_swap_transformation,
            self._random_instruction_transformation,
        )
        index = np.random.choice(
            range(len(generators)),
            1,
            p=self._normalized_weights,
        )[0]
        random_transformation = generators[index](program)
        prob = self._normalized_weights[index]

        if random_transformation is None:
            # Unable to do transformation, return itself
            return program, prob, prob

        # The fwd and bwd probabilities are equal
        # because the inverses of these transformations
        # are of the same class of transformations
        candidate = transformation_eval(random_transformation, program)
        return candidate, prob, prob

    def _verification(self, program: Program) -> float:
        """Calculate verification score for test cases"""

        program_string = json.dumps(program)
        if (result := self._verification_cache.get(program_string)) is None:
            result = z3_prove_equivalence_or_find_counterexample(
                self._initial_program, program
            )
            self._verification_cache[program_string] = result

        if result.counterexample is not None:
            # Z3 found a counterexample, add it to test cases
            self.test_cases.add(
                (
                    tuple(result.counterexample.arguments1),
                    cast(
                        int, result.counterexample.return1
                    ),  # TODO cover other than int
                )
            )

        if (
            result.argument_type_equivalent
            and result.arity_equivalent
            and result.return_equivalent
        ):
            return 0.0
        return 1.0

    def cost(self, program: Program) -> Probability:
        """Calculate the MCMC cost function"""
        equivalence_cost = calculate_validation(self.brili, program, self.test_cases)

        if equivalence_cost == 0:
            equivalence_cost = self._verification(program)
        logger.debug(equivalence_cost)

        if self.performance_correctness_ratio == 0:
            return equivalence_cost

        performance_cost = calculate_performance(program)

        return (
            100 * equivalence_cost
            + self.performance_correctness_ratio * performance_cost
        )


def bloke(brili, program: Program, beta: float) -> Program:
    sampler = BlokeSample(
        brili,
        program,
        opcode_weight=1,
        operand_weight=1,
        swap_weight=1,
        instruction_weight=1,
        unused_probability=None,
        mcmc_beta=beta,
    )

    sampler.performance_correctness_ratio = 0.01
    maximum_ratio = 0.95

    best_program: Program = program

    log_interval = 1000
    i = 0

    t0 = time.time()
    while sampler.performance_correctness_ratio < maximum_ratio:
        best_cost: float = sampler.cost(best_program)
        for _ in range(10000):
            program, cost = sampler.sample(program, best_cost)
            if cost <= best_cost:
                best_program, best_cost = program, cost
            if i % log_interval == 0:
                t1 = time.time()
                logger.info(
                    f"ITERATION:   {i}\n"
                    f"RATIO:       {sampler.performance_correctness_ratio}\n"
                    f"BEST_COST:   {best_cost}\n"
                    f"TEST_CASES:  {len(sampler.test_cases)}\n"
                    f"PERFORMANCE: {log_interval / (t1 - t0)}"
                )
                logger.info(prints_prog(best_program))
                t0 = t1
            i += 1
        sampler.performance_correctness_ratio += 0.10

        if i > log_interval * 100:
            break

    return best_program


BRILI_MAP: dict[str, Brili] = {
    "subprocess": SubprocessBrili(),
    "brilirs": Brilirs(),
}


@click.command()
@click.option(
    "--brili",
    default="subprocess",
    type=click.Choice(BRILI_MAP.keys(), case_sensitive=False),
)
@click.option(
    "--beta",
    type=float,
    help="Beta value for MCMC",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    type=bool,
    help="Verbose output",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    type=bool,
    help="Debug output",
)
def main(brili: str, beta: float, verbose: bool, debug: bool) -> None:
    brili_impl = BRILI_MAP[brili]

    logging_config: dict[str, Any] = {
        "stream": sys.stderr,
        "encoding": "utf-8",
        "format": "%(message)s",
    }

    if verbose:
        logging_config["level"] = logging.INFO
    if debug:
        logging_config["level"] = logging.DEBUG

    logging.basicConfig(**logging_config)

    # program: Program = json.load(sys.stdin)
    program: Program = json.loads(
        """{"functions":[{"args":[{"name":"a","type":"int"},{"name":"b","type":"int"},{"name":"c","type":"int"}],"instrs":[{"args":["a","b"],"dest":"x1","op":"mul","type":"int"},{"args":["a","c"],"dest":"x2","op":"mul","type":"int"},{"args":["x1","x2"],"dest":"x3","op":"add","type":"int"},{"args":["x3"],"op":"ret"}],"name":"main","type":"int"}]}"""
    )
    optimized_program = bloke(brili_impl, program, beta)
    print(json.dumps(optimized_program))


if __name__ == "__main__":
    main()
