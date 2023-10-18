"""Relabels all variables with a label"""

import copy
import json
import sys
from typing import cast

import click

from .typing_bril import Effect, Program, Value, Variable


def renamed_variable_get(var: Variable, label: int | str) -> Variable:
    """Get new renamed variable"""
    return Variable(f"{var}.{label}")


def rename_variables_in_program(program: Program, label: int | str) -> Program:
    """Rename all variables in a program with a label"""
    new_program = copy.deepcopy(program)
    for func in new_program["functions"]:
        if "args" in func:
            for i, arg in enumerate(func["args"]):
                func["args"][i]["name"] = renamed_variable_get(arg["name"], label)
        for instruction in func["instrs"]:
            if "dest" in instruction:
                value = cast(Value, instruction)
                value["dest"] = renamed_variable_get(value["dest"], label)
            if "args" in instruction:
                effect = cast(Effect, instruction)
                for i, var in enumerate(effect["args"]):
                    effect["args"][i] = renamed_variable_get(var, label)

    return new_program


@click.command()
@click.option(
    "-l",
    "--label",
    default="0",
    type=str,
    help="label",
)
def main(label: str) -> None:
    program: Program = json.load(sys.stdin)
    new_program = rename_variables_in_program(program, label)
    print(json.dumps(new_program))


if __name__ == "__main__":
    main()
