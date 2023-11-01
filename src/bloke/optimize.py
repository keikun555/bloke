"""Bril optimizer using STOKE"""

import copy
import json
import sys
from collections import defaultdict
from typing import Callable, TypeAlias, cast

import click
import numpy as np
import numpy.typing as npt

from bloke.bril_equivalence import (
    briltxt_get,
    z3_prove_equivalence_or_find_counterexample,
)
from bloke.mcmc import MonteCarloMarkovChainSample, Probability
from bril.bril2z3 import COMPATIBLE_OPS
from bril.bril_constants import OPERATORS, GenericType
from bril.brili import brili
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


def one_testcase_validation(program: Program, testcase: TestCase) -> float:
    """Calculate validation score for one test case"""
    arguments, expected_output = testcase
    result = brili(program, arguments, profile=True)
    if result.error is not None:
        # In the future we may want to give different kinds of scores for different errors
        return 1.0

    return np.log(1 + float(abs(result.returncode - expected_output)))


def calculate_validation(program: Program, test_cases: list[TestCase]) -> float:
    """Calculate validation score for test cases"""
    return sum(one_testcase_validation(program, test_case) for test_case in test_cases)


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


class BlokeSample(MonteCarloMarkovChainSample[Program]):
    """Does MCMC on Bril programs"""

    def __init__(
        self,
        initial_program: Program,
        opcode_weight: int,
        operand_weight: int,
        swap_weight: int,
        instruction_weight: int,
        unused_probability: Probability,
        mcmc_beta: float,
    ):
        super().__init__(mcmc_beta)

        self._initial_program = initial_program

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

        self._unused_probability: Probability = unused_probability

        # For the cost function
        self.test_cases: list[TestCase] = []
        self.performance_correctness_ratio: float = 0.0

    def _random_instruction_in_program(
        self, program: Program, qualifier: Callable[[Instruction], bool] | None = None
    ) -> tuple[str, int, Instruction] | None:
        instructions: list[tuple[str, int, Instruction]] = []
        for function in program["functions"]:
            for i, instruction in enumerate(function["instrs"]):
                if qualifier is not None and not qualifier(instruction):
                    continue
                instructions.append((function["name"], i, instruction))

        if len(instructions) <= 0:
            return None

        return instructions[np.random.choice(len(instructions))]

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
        random_instruction = self._random_instruction_in_program(
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
        random_instruction = self._random_instruction_in_program(
            program, lambda i: "args" in i and len(cast(Effect, i)["args"]) > 0
        )
        if random_instruction is None:
            return None
        (
            function_name,
            instruction_index,
            instruction,
        ) = random_instruction

        variable_to_type, type_to_variables = self._variable_type_dicts_get(
            program, function_name
        )

        effect = cast(Effect, instruction)

        try:
            operand_index = np.random.choice(len(effect["args"]))
            operand_type = variable_to_type[effect["args"][operand_index]]
            random_variable = np.random.choice(type_to_variables[operand_type])
        except KeyError as e:
            print("failed")
            print(effect)
            sys.stdin = open("/dev/tty")
            breakpoint()
            raise e

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
        random_instruction1 = self._random_instruction_in_program(program)
        random_instruction2 = self._random_instruction_in_program(program)
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
        random_instruction_to_replace = self._random_instruction_in_program(
            program, lambda i: "op" in i
        )
        if random_instruction_to_replace is None:
            return None
        (
            function_name,
            instruction_index,
            instruction,
        ) = random_instruction_to_replace

        variable_to_type: dict[Variable, BrilType] = {}
        type_to_variables: dict[BrilType, list[Variable]] = defaultdict(list)
        for function in program["functions"]:
            if function["name"] != function_name:
                continue
            for instruction in function["instrs"]:
                if "dest" in instruction:
                    value = cast(Value, instruction)
                    variable_to_type[value["dest"]] = value["type"]
                    type_to_variables[value["type"]].append(value["dest"])
            break

        random_instruction: Instruction

        if np.random.rand() < self._unused_probability:
            random_instruction = {"op": "nop"}
        else:

            def filter_(operation: Operation) -> bool:
                if operation in ("nop", "phi", "const"):
                    return False

                argument_types, _ = self._operation_dict[operation]
                for type_ in argument_types:
                    if (
                        type_ != "generic"
                        and len(type_to_variables[cast(BrilType, type_)]) > 0
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
            random_operation = {"op": random_operation}

            argument_types, return_type = self._operation_dict[random_operation]
            arguments: list[Variable] = []
            generic_type: BrilType = variable_to_type[
                np.random.choice(list(variable_to_type.keys()))
            ]
            for type_ in argument_types:
                if type_ == "generic":
                    variables = type_to_variables[generic_type]
                else:
                    variables = type_to_variables[cast(BrilType, type_)]
                random_operand = np.random.choice(variables)
                arguments.append(random_operand)

            if len(arguments) > 0:
                random_operation["args"] = arguments

            if return_type is not None:
                random_operation["type"] = return_type

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
        try:
            random_transformation = generators[index](program)
        except KeyError as e:
            print("failed")
            briltxt = briltxt_get()
            briltxt.print_prog(program)
            raise e
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
        result = z3_prove_equivalence_or_find_counterexample(
            self._initial_program, program
        )

        if result.counterexample is not None:
            # Z3 found a counterexample, add it to test cases
            self.test_cases.append(
                (
                    tuple(result.counterexample.arguments1),
                    cast(int, result.counterexample.return1),
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
        equivalence_score = calculate_validation(program, self.test_cases)

        if equivalence_score == 0:
            equivalence_score = self._verification(program)

        if self.performance_correctness_ratio == 0:
            return equivalence_score

        performance_score = calculate_performance(program)

        return (
            equivalence_score + self.performance_correctness_ratio * performance_score
        )


def bloke(program: Program, beta: float) -> Program:
    sampler = BlokeSample(
        program,
        opcode_weight=1,
        operand_weight=1,
        swap_weight=1,
        instruction_weight=1,
        unused_probability=1,
        mcmc_beta=beta,
    )

    sampler.performance_correctness_ratio = 0.01
    maximum_ratio = 0.95

    best_program: Program = program

    log_interval = 100
    i = 0
    briltxt = briltxt_get()

    while sampler.performance_correctness_ratio < maximum_ratio:
        best_cost: float = sampler.cost(best_program)
        for _ in range(100):
            candidate, cost = sampler.sample(best_program, best_cost)
            if cost < best_cost:
                best_program, best_cost = candidate, cost
            if i % log_interval == 0:
                print(f"ITERATION: {i}")
                print(f"RATIO:     {sampler.performance_correctness_ratio}")
                print(f"BEST_COST: {best_cost}")
                briltxt.print_prog(best_program)
            i += 1
        sampler.performance_correctness_ratio *= 1.01

    return best_program


@click.command()
@click.option(
    "--beta",
    type=float,
    help="Beta value for MCMC",
)
def main(beta: float) -> None:
    program: Program = json.load(sys.stdin)
    optimized_program = bloke(program, beta)
    print(json.dumps(optimized_program))


if __name__ == "__main__":
    main()
