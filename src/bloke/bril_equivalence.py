"""Equivalence of Bril programs"""
import copy
import importlib
import importlib.util
import json
import operator
import subprocess
import sys
from types import MappingProxyType, ModuleType
from typing import Callable, Generator, NamedTuple, TypeAlias, TypedDict, TypeVar, cast

import click
import z3

from bril import bril2z3
from bril.basic_blocks import (
    BasicBlock,
    BasicBlockFunction,
    BasicBlockProgram,
    basic_block_program_from_program,
)
from bril.bril_extract import phi_nodes_get, var_to_type_dict_get
from bril.bril_labeler import index_to_label_dict_get, label_to_index_dict_get
from bril.cfg import control_flow_graph_from_instructions, is_cyclic
from bril.typing_bril import (
    BrilType,
    Constant,
    Effect,
    Instruction,
    Operation,
    Program,
    Value,
    Variable,
)
from bril.bril_variable_labeler import rename_variables_in_program


def briltxt_get() -> ModuleType:
    def getGitRoot():
        return (
            subprocess.Popen(
                ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE
            )
            .communicate()[0]
            .rstrip()
            .decode("utf-8")
        )

    full_path = f"{getGitRoot()}/vendor/bril/bril-txt/briltxt.py"

    spec = importlib.util.spec_from_file_location("briltxt", full_path)
    if spec is None:
        raise Exception(f"briltxt not found at {full_path}")

    briltxt = importlib.util.module_from_spec(spec)
    sys.modules["briltxt"] = briltxt

    if spec.loader is None:
        raise Exception(f"briltxt loader not found at {full_path}")
    spec.loader.exec_module(briltxt)

    return briltxt


def z3_prove_equivalence_or_find_counterexample(
    program1: Program, program2: Program
) -> z3.ExprRef:

    program1 = rename_variables_in_program(program1, 0)
    program2 = rename_variables_in_program(program2, 1)
    bb_program1: BasicBlockProgram = basic_block_program_from_program(program1)
    bb_program2: BasicBlockProgram = basic_block_program_from_program(program2)

    expression1 = bril2z3.program_to_z3(bb_program1, program_label=0)
    expression2 = bril2z3.program_to_z3(bb_program2, program_label=1)
    print(expression1, file=sys.stderr)
    print(expression2, file=sys.stderr)

    solver = z3.Solver()
    solver.add(expression1)
    solver.add(expression2)
    solver.add()
    print(solver, file=sys.stderr)
    return expression1


@click.command()
@click.argument("program1_filepath", type=click.Path(exists=True))
@click.argument("program2_filepath", type=click.Path(exists=True))
def main(program1_filepath: str, program2_filepath: str) -> None:
    briltxt = briltxt_get()

    with open(program1_filepath, "r", encoding="utf-8") as file:
        program1: Program = json.loads(briltxt.parse_bril(file.read()))
    with open(program2_filepath, "r", encoding="utf-8") as file:
        program2: Program = json.loads(briltxt.parse_bril(file.read()))

    something = z3_prove_equivalence_or_find_counterexample(program1, program2)


if __name__ == "__main__":
    main()
