"""Extract relevant information from Basic Block Programs"""

from typing import Optional, cast

from bril.basic_blocks import BasicBlock, BasicBlockFunction, BasicBlockProgram
from bril.bril_analyze import has_label, is_value
from bril.typing_bril import Argument, BrilType, Label, Value, Variable


def function_arguments_get(function: BasicBlockFunction) -> list[Argument]:
    """Return list of arguments of a function"""
    if "args" not in function:
        return []
    return function["args"]


def main_function_get(program: BasicBlockProgram) -> BasicBlockFunction | None:
    """Return main function if it exists"""
    for func in program["functions"]:
        if func["name"] == "main":
            return func

    return None


def label_get(basic_block: BasicBlock) -> Optional[str]:
    """Given a basic block get its label or None if the block doesn't have one"""
    if not has_label(basic_block):
        return None
    label = cast(Label, basic_block[0])
    return label["label"]


def values_get(basic_block: BasicBlock) -> tuple[Value, ...]:
    """Given a basic block get its defined values"""
    values: list[Value] = []

    for instruction in basic_block:
        if is_value(instruction):
            values.append(cast(Value, instruction))

    return tuple(values)


def phi_nodes_get(basic_block: BasicBlock) -> tuple[Value, ...]:
    """Given an SSA basic block get its phi nodes"""
    phi_nodes: list[Value] = []

    for instruction in basic_block:
        if "op" in instruction:
            ssa_instruction = cast(Value, instruction)
            if ssa_instruction["op"] == "phi":
                phi_nodes.append(ssa_instruction)

    return tuple(phi_nodes)


def var_to_type_dict_get(func: BasicBlockFunction) -> dict[Variable, BrilType]:
    """Given function return variable to bril type dictionary"""
    var_to_type: dict[Variable, BrilType] = {}

    # Function arguments
    if "args" in func:
        for argument in func["args"]:
            var = Variable(
                argument["name"]
            )  # not sure why we need to encapsulate with Variable
            var_to_type[var] = argument["type"]

    # Now Value instructions
    for block in func["instrs"]:
        for value in values_get(block):
            var = value["dest"]
            var_to_type[var] = value["type"]

    return var_to_type
