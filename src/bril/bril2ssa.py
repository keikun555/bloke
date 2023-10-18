"""
Converts Bril programs to SSA form
"""

import sys
import json
import copy

from typing import cast, Generator
from collections import defaultdict

import click

from typing_bril import (
    Program,
    BrilType,
    Variable,
    Effect,
    Value,
    Instruction,
)
from basic_blocks import (
    BasicBlock,
    BasicBlockFunction,
    BasicBlockProgram,
    basic_block_program_from_program,
    program_from_basic_block_program,
)
from cfg import (
    ControlFlowGraph,
    control_flow_graph_from_instructions,
)

from bril_labeler import index_to_label_dict_get, apply_labels

from bril_extract import values_get, phi_nodes_get

from dominance_analysis import (
    dominance_frontier_indices_get,
    immediate_dominator_index_get,
    index_dominator_tree_get,
    DominanceAnalysis,
)


def new_variable_generator(variable: Variable) -> Generator[Variable, None, None]:
    i = 0
    while True:
        yield Variable(f"{variable}.{i}")
        i += 1


def add_entry_block_with_function_arguments(
    bb_func: BasicBlockFunction,
) -> BasicBlockFunction:
    """Modify basic block function to explicitly state function arguments in a new entry block"""
    new_bb_func = copy.deepcopy(bb_func)

    entry_block: list[Instruction] = []
    if "args" in bb_func:
        for argument in bb_func["args"]:
            var: Variable = Variable(
                argument["name"]
            )  # not sure why we need to encapsulate with Variable
            value = Value(op="id", dest=var, type=argument["type"], args=[var])
            entry_block.append(value)

    if len(entry_block) > 0:
        new_bb_func["instrs"].insert(0, entry_block)

    return new_bb_func


def defs_orig_variable_dicts_get(
    bb_func: BasicBlockFunction,
) -> tuple[
    dict[Variable, set[int]], dict[int, set[Variable]], dict[Variable, BrilType]
]:

    defs: dict[Variable, set[int]] = defaultdict(set)
    orig: dict[int, set[Variable]] = defaultdict(set)
    var_to_type: dict[Variable, BrilType] = {}

    # Function arguments
    if "args" in bb_func:
        for argument in bb_func["args"]:
            var = Variable(
                argument["name"]
            )  # not sure why we need to encapsulate with Variable
            defs[var].add(0)
            orig[0].add(var)
            var_to_type[var] = argument["type"]

    # Now Value instructions
    for i, block in enumerate(bb_func["instrs"]):
        for value in values_get(block):
            var = value["dest"]
            defs[var].add(i)
            orig[i].add(var)
            var_to_type[var] = value["type"]

    return defs, orig, var_to_type


def insert_phi_nodes(
    bb_func: BasicBlockFunction,
    defs: dict[Variable, set[int]],
    orig: dict[int, set[Variable]],
    var_to_type: dict[Variable, BrilType],
) -> BasicBlockFunction:
    """Insert Phi nodes and return SSA basic block function"""
    cfg: ControlFlowGraph = control_flow_graph_from_instructions(bb_func["instrs"])
    dominance_frontier_indices = dominance_frontier_indices_get(cfg)

    ssa_bb_function: BasicBlockFunction = cast(
        BasicBlockFunction, copy.deepcopy(bb_func)
    )
    ssa_basic_blocks: list[BasicBlock] = ssa_bb_function["instrs"]
    phi_dests: dict[Variable, set[int]] = defaultdict(set)
    for v in sorted(var_to_type.keys()):
        v_type = var_to_type[v]
        v_defs = sorted(list(defs[v]))

        while len(v_defs) > 0:
            d = v_defs.pop()

            for frontier_index in sorted(list(dominance_frontier_indices[d])):
                if not 0 <= frontier_index < len(ssa_basic_blocks):
                    continue

                # add a phi node to block unless we have done so already
                if frontier_index not in phi_dests[v]:
                    phi_node = Value(op="phi", dest=v, type=v_type, args=[], labels=[])
                    ssa_basic_blocks[frontier_index].insert(1, phi_node)
                    phi_dests[v].add(frontier_index)

                    # add block to v_defs because it now writes to v
                    if v not in orig[frontier_index]:
                        v_defs.append(frontier_index)

    return ssa_bb_function


def bb_func_to_ssa_bb_func(
    bb_func: BasicBlockFunction,
) -> BasicBlockFunction:
    """Given basic blocks to a function and its CFG compute the SSA version of the basic blocks"""
    bb_func = add_entry_block_with_function_arguments(bb_func)

    # Add labels
    index_to_label = index_to_label_dict_get(bb_func)
    bb_func["instrs"] = apply_labels(bb_func["instrs"], index_to_label)

    defs, orig, var_to_type = defs_orig_variable_dicts_get(bb_func)

    # Insert phi nodes
    ssa_bb_function = insert_phi_nodes(bb_func, defs, orig, var_to_type)
    ssa_basic_blocks: list[BasicBlock] = ssa_bb_function["instrs"]

    cfg = control_flow_graph_from_instructions(cast(list[BasicBlock], ssa_basic_blocks))

    # Rename variables
    variable_dict: dict[
        Variable,
        tuple[
            list[
                tuple[Variable, int]
            ],  # stack of variable names and indices they are defined in
            Generator[Variable, None, None],  # generates new variables
            set[Variable],  # new variables derived from v
        ],
    ] = {var: ([], new_variable_generator(var), set()) for var in var_to_type}

    # Function arguments
    if "args" in ssa_bb_function:
        for argument in ssa_bb_function["args"]:
            old_variable = Variable(
                argument["name"]
            )  # not sure why we need to encapsulate with Variable
            stack, generator, _ = variable_dict[old_variable]
            argument["name"] = next(generator)
            stack.append((Variable(argument["name"]), 0))

    dominator_tree = index_dominator_tree_get(cfg)

    def determine_phi_label_index(
        immediate_dominator_index: DominanceAnalysis,
        phi_label_index_candidates: set[int],
        predecessor_index: int,
    ) -> int | None:
        index = predecessor_index
        while index not in phi_label_index_candidates:
            if len(immediate_dominator_index[index]) <= 0:
                return None
            index = next(iter(immediate_dominator_index[index]))
        return index

    def rename(block_index: int):
        block = ssa_basic_blocks[block_index]

        block_variables: set[Variable] = set()
        for instr in block:
            # Replace each argument to instr with stack_[old name]
            if "args" in instr:
                effect = cast(Effect, instr)
                for i, old_variable in enumerate(effect["args"]):
                    if old_variable not in variable_dict:
                        continue
                    stack, _, _ = variable_dict[old_variable]
                    effect["args"][i], _ = stack[-1]

            # Replace instr's destination with a new name
            if "dest" in instr:
                value = cast(Value, instr)
                old_variable = value["dest"]
                stack, generator, new_variables = variable_dict[old_variable]

                value["dest"] = next(generator)
                new_variables.add(value["dest"])

                if old_variable in block_variables:
                    # if already defined then replace
                    stack[-1] = (value["dest"], block_index)
                else:
                    stack.append((value["dest"], block_index))

                block_variables.add(old_variable)

        for s in cfg.successors(block_index):
            if not 0 <= s < len(ssa_basic_blocks):
                continue
            for phi_node in phi_nodes_get(ssa_basic_blocks[s]):
                variable = phi_node["dest"]

                if variable in variable_dict:
                    old_variable = variable
                else:
                    old_variable = next(
                        var
                        for var, (_, _, new_variables) in variable_dict.items()
                        if variable in new_variables
                    )

                stack, _, _ = variable_dict[old_variable]

                phi_label_index_to_arg_dict: dict[int, Variable] = {
                    index: arg for arg, index in stack
                }
                phi_label_index_candidates = set(index for _, index in stack)
                immediate_dominator_index = immediate_dominator_index_get(cfg)
                if (
                    phi_label_index := determine_phi_label_index(
                        immediate_dominator_index,
                        phi_label_index_candidates,
                        block_index,
                    )
                ) is not None:
                    phi_node["args"].append(
                        phi_label_index_to_arg_dict[phi_label_index]
                    )
                    phi_node["labels"].append(index_to_label[block_index])

        for isub in dominator_tree[block_index]:
            if not 0 <= isub < len(ssa_basic_blocks):
                continue
            rename(isub)

        while len(block_variables) > 0:
            old_variable = block_variables.pop()
            stack, _, _ = variable_dict[old_variable]
            stack.pop()

    if len(ssa_basic_blocks) > 0:
        rename(0)

    return ssa_bb_function


def bril_to_ssa(program: Program) -> Program:
    # Says Program is not Program
    bb_program: BasicBlockProgram = basic_block_program_from_program(program)  # type: ignore
    ssa_bb_program: BasicBlockProgram = cast(
        BasicBlockProgram, copy.deepcopy(bb_program)
    )

    for i, func in enumerate(bb_program["functions"]):
        ssa_bb_program["functions"][i] = bb_func_to_ssa_bb_func(func)

    return program_from_basic_block_program(ssa_bb_program)


@click.command()
def main():
    prog: Program = json.load(sys.stdin)
    ssa_program: Program = bril_to_ssa(prog)
    print(json.dumps(ssa_program))


if __name__ == "__main__":
    main()
