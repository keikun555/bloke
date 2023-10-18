"""
Converts SSA form Bril programs to non-SSA form
"""

import copy
import json
import sys
from collections import defaultdict
from typing import cast

import click

from basic_blocks import (BasicBlock, BasicBlockFunction, BasicBlockProgram,
                          basic_block_program_from_program,
                          program_from_basic_block_program)
from bril_analyze import is_terminator
from bril_extract import phi_nodes_get
from bril_labeler import apply_labels, index_to_label_dict_get
from cfg import control_flow_graph_from_instructions
from typing_bril import Constant, Effect, Instruction, Program, Value, Variable


def ssa_bb_func_to_bb_func(
    ssa_bb_func: BasicBlockFunction,
) -> BasicBlockFunction:
    """Given SSA basic blocks to a function compute the non-SSA basic blocks"""
    ssa_bb_func = copy.deepcopy(ssa_bb_func)

    index_to_label = index_to_label_dict_get(cast(BasicBlockFunction, ssa_bb_func))
    ssa_bb_func["instrs"] = cast(
        list[BasicBlock],
        apply_labels(cast(BasicBlockFunction, ssa_bb_func)["instrs"], index_to_label),
    )

    cfg = control_flow_graph_from_instructions(
        cast(list[BasicBlock], ssa_bb_func["instrs"])
    )

    for block_index, ssa_basic_block in enumerate(ssa_bb_func["instrs"]):
        phi_nodes = phi_nodes_get(ssa_basic_block)

        for phi_node in phi_nodes:
            dest = phi_node["dest"]
            type_ = phi_node["type"]

            for predecessor_index in cfg.predecessors(block_index):
                predecessor_label = index_to_label[predecessor_index]
                instructions: list[Instruction] = []
                try:
                    arg = next(
                        arg
                        for arg, label in zip(phi_node["args"], phi_node["labels"])
                        if label == predecessor_label
                    )
                    instructions.append(
                        Value(op="id", type=type_, dest=dest, args=[arg])
                    )
                except StopIteration:
                    # Not in Phi node
                    if isinstance(type_, dict):
                        if "ptr" in type_:
                            instructions.append(
                                Constant(
                                    op="const",
                                    type="int",
                                    dest=Variable(f"{dest}.size"),
                                    value=1,
                                )
                            )
                            instructions.append(
                                Value(
                                    op="alloc",
                                    args=[Variable(f"{dest}.size")],
                                    dest=dest,
                                    type=type_,
                                )
                            )
                            instructions.append(Effect(op="free", args=[dest]))
                    else:
                        constant_value: int | float | bool
                        match type_:
                            case "int":
                                constant_value = int()
                            case "float":
                                constant_value = float()
                            case "bool":
                                constant_value = bool()
                            case _:
                                raise ValueError(
                                    f"Encountered unsupported type: {type_}"
                                )

                        instructions.append(
                            Constant(
                                op="const", type=type_, dest=dest, value=constant_value
                            )
                        )

                block_to_insert = ssa_bb_func["instrs"][predecessor_index]
                if len(block_to_insert) <= 0 or not is_terminator(block_to_insert[-1]):
                    block_to_insert.extend(instructions)
                else:
                    for instruction in instructions:
                        block_to_insert.insert(-1, instruction)

    for block_index, ssa_basic_block in enumerate(ssa_bb_func["instrs"]):
        # Delete phi nodes
        ssa_bb_func["instrs"][block_index] = list(
            filter(
                lambda instr: "op" not in instr
                or ("op" in instr and cast(Value, instr)["op"] != "phi"),
                ssa_basic_block,
            )
        )

    bb_function = cast(BasicBlockFunction, ssa_bb_func)

    return bb_function


def ssa_to_bril(ssa_program: Program) -> Program:
    # Says Program is not Program
    ssa_bb_program: BasicBlockProgram = basic_block_program_from_program(ssa_program)
    bb_program: BasicBlockProgram = cast(
        BasicBlockProgram, copy.deepcopy(ssa_bb_program)
    )

    for i, func in enumerate(ssa_bb_program["functions"]):
        bb_program["functions"][i] = ssa_bb_func_to_bb_func(func)

    return cast(Program, program_from_basic_block_program(bb_program))


@click.command()
def main():
    ssa_program: Program = json.load(sys.stdin)
    program: Program = ssa_to_bril(ssa_program)
    print(json.dumps(program))


if __name__ == "__main__":
    main()
