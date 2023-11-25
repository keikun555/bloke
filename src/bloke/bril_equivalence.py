"""Equivalence of Bril programs"""
import json
import logging
from dataclasses import asdict, dataclass
from typing import cast

import briltxt
import click
import numpy as np
import z3

from bril import bril2z3
from bril.basic_blocks import BasicBlockProgram, basic_block_program_from_program
from bril.bril2ssa import bril_to_ssa
from bril.bril_extract import function_arguments_get, main_function_get
from bril.bril_variable_labeler import rename_variables_in_program
from bril.typing_bril import PrimitiveType, Program

logger = logging.getLogger(__name__)


def to_signed(unsigned_integer: int) -> int:
    """Convert an unsigned integer to its signed form"""
    return int(np.int64(np.uint64(unsigned_integer)))


def z3_value_to_python_value(
    value: z3.ExprRef, ctx: z3.Context | None = None
) -> PrimitiveType | None:
    """Convert Z3 value to python value"""
    if isinstance(value, z3.BitVecNumRef):
        return to_signed(cast(z3.BitVecNumRef, value).as_long())
    if isinstance(value, z3.FPRef):
        fraction = z3.RealVal(value).as_fraction()
        return float(fraction.numerator / fraction.denominator)
    if isinstance(value, z3.BoolRef):
        return cast(z3.BoolRef, value).__bool__()
    if isinstance(value, z3.DatatypeRef):
        if value == bril2z3.z3_bril_any_type(ctx=ctx).nil:
            return None
        return z3_value_to_python_value(value.children()[0], ctx=ctx)
    raise Exception(f"z3_value_to_python_value, type {type(value)} not implemented")


@dataclass
class Counterexample:
    """Counterexample dataclass"""

    arguments1: list[int | float | bool]
    arguments2: list[int | float | bool]
    return1: int | float | bool | None
    return2: int | float | bool | None


@dataclass
class EquivalenceAnalysisResult:
    """Datastructure to house result of equivalence analysis"""

    arity_equivalent: bool
    argument_type_equivalent: bool
    return_equivalent: bool
    # if arity and argument types are equivalent and return is not equivalent, produce a model
    counterexample: Counterexample | None


def program_to_equivalence_analysis_compatible_program(
    program: Program, label: int | str
) -> BasicBlockProgram | None:
    """
    Make a program equivalence analysis compatible program
    The variables of a compatible program are relabeled
    The program is in SSA form
    The program is a basic block program
    """
    try:
        program = rename_variables_in_program(program, label)
        ssa_program = bril_to_ssa(program)
        bb_program = basic_block_program_from_program(ssa_program)

        return bb_program
    except IndexError:
        # Unable to convert to equivalence analysis compatible program
        return None


def z3_prove_equivalence_or_find_counterexample(
    program1: Program, program2: Program
) -> EquivalenceAnalysisResult:
    """Do an equivalence analysis of two programs and return equivalence analysis result"""

    result = EquivalenceAnalysisResult(False, False, False, None)

    bb_program1 = program_to_equivalence_analysis_compatible_program(program1, 0)
    bb_program2 = program_to_equivalence_analysis_compatible_program(program2, 1)

    if bb_program1 is None or bb_program2 is None:
        return result

    context = z3.Context()
    solver = z3.Solver(ctx=context)

    # Make sure arguments are equivalent
    main_function1 = main_function_get(bb_program1)
    function_arguments1 = (
        function_arguments_get(main_function1) if main_function1 is not None else []
    )
    main_function2 = main_function_get(bb_program2)
    function_arguments2 = (
        function_arguments_get(main_function2) if main_function2 is not None else []
    )
    result.arity_equivalent = len(function_arguments1) == len(function_arguments2)
    result.argument_type_equivalent = all(
        arg1["type"] == arg2["type"]
        for arg1, arg2 in zip(function_arguments1, function_arguments2)
    )
    if not result.arity_equivalent or not result.argument_type_equivalent:
        # Short circuit
        return result

    for arg1, arg2 in zip(function_arguments1, function_arguments2):
        solver.add(
            bril2z3.z3_bril_argument_get(arg1, ctx=context)
            == bril2z3.z3_bril_argument_get(arg2, ctx=context)
        )

    # We check satisfiability of the returns being different
    # Does there exist inputs such that the returns are different?
    return1 = bril2z3.bril_ret_var_to_z3(0, ctx=context)
    return2 = bril2z3.bril_ret_var_to_z3(1, ctx=context)
    solver.add(return1 != return2)

    # Lift programs to Z3
    try:
        expression1 = bril2z3.program_to_z3(bb_program1, program_label=0, ctx=context)
        expression2 = bril2z3.program_to_z3(bb_program2, program_label=1, ctx=context)
    except KeyError:
        # Unable to convert to Z3
        return result

    solver.add(expression1)
    solver.add(expression2)

    # Run the solver
    z3_result = solver.check()

    if z3_result == z3.unsat:
        # Programs are equivalent
        result.return_equivalent = True
    elif z3_result == z3.sat:
        # Counterexample found
        result.return_equivalent = False
        model = solver.model()
        result.counterexample = Counterexample([], [], None, None)
        for arg in function_arguments1:
            python_argument = z3_value_to_python_value(
                model[bril2z3.z3_bril_argument_get(arg, ctx=context)],
                ctx=context,
            )
            result.counterexample.arguments1.append(python_argument)
        for arg in function_arguments2:
            python_argument = z3_value_to_python_value(
                model[bril2z3.z3_bril_argument_get(arg, ctx=context)],
                ctx=context,
            )
            result.counterexample.arguments2.append(python_argument)

        result.counterexample.return1 = z3_value_to_python_value(
            model[return1],
            ctx=context,
        )
        result.counterexample.return2 = z3_value_to_python_value(
            model[return2],
            ctx=context,
        )

    return result


@click.command()
@click.argument("program1_filepath", type=click.Path(exists=True))
@click.argument("program2_filepath", type=click.Path(exists=True))
def main(program1_filepath: str, program2_filepath: str) -> None:
    with open(program1_filepath, "r", encoding="utf-8") as file:
        program1: Program = json.loads(briltxt.parse_bril(file.read()))
    with open(program2_filepath, "r", encoding="utf-8") as file:
        program2: Program = json.loads(briltxt.parse_bril(file.read()))

    result = z3_prove_equivalence_or_find_counterexample(program1, program2)

    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
