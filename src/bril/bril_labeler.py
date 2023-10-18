"""Creates labels for CFG blocks if it doesn't exist"""

import copy
import json
import sys
from typing import Generator, Optional, TypeAlias

import click

from basic_blocks import (BasicBlock, BasicBlockFunction, BasicBlockProgram,
                          basic_block_program_from_program,
                          program_from_basic_block_program)
from bril_analyze import has_label
from bril_extract import label_get
from cfg import ControlFlowGraph, control_flow_graph_from_instructions
from typing_bril import Label, Program

ENTRY_BLOCK_LABEL = "ENTRY"
EXIT_BLOCK_LABEL = "EXIT"

IndexToLabelDict: TypeAlias = dict[int, str]


def unique_label_name_generator(
    func_name: str, used_labels: set[str]
) -> Generator[str, None, None]:
    """Generates new labels"""
    i = 0
    while True:
        candidate = f"{func_name}.b{i}"
        if candidate not in used_labels:
            yield candidate
            used_labels.add(candidate)
        i += 1


def is_labeled(basic_blocks: list[BasicBlock]) -> bool:
    """Returns whether basic blocks are labeled"""
    for basic_block in basic_blocks:
        if label_get(basic_block) is None:
            return False

    return True


def index_to_label_dict_get(
    basic_block_function: BasicBlockFunction, cfg: Optional[ControlFlowGraph] = None
) -> IndexToLabelDict:
    """
    Assigns basic block indices to labels.
    Creates new labels if a block does not have a label.
    If optional CFG is provided, creates entries for virtual entry and exit blocks
    Does not modify the code.

    DOES NOT ACCOUNT FOR CROSS-FUNCTION LABEL NAME CLASHES SO BEWARE!
    """
    index_to_label: IndexToLabelDict = {}
    no_label_indices = []
    basic_blocks = basic_block_function["instrs"]
    for i, basic_block in enumerate(basic_blocks):
        label = label_get(basic_block)
        if label is not None:
            index_to_label[i] = label
        else:
            no_label_indices.append(i)

    label_name_generator = unique_label_name_generator(
        basic_block_function["name"], set(index_to_label.values())
    )
    for i in no_label_indices:
        index_to_label[i] = next(label_name_generator)

    if cfg is not None:
        index_to_label[cfg.entry] = ENTRY_BLOCK_LABEL
        index_to_label[cfg.exit] = EXIT_BLOCK_LABEL

    return index_to_label

LabelToIndexDict: TypeAlias = dict[str, int]

def label_to_index_dict_get(index_to_label_dict: IndexToLabelDict) -> LabelToIndexDict:
    """Flips index and labels"""
    label_to_index: LabelToIndexDict = {}

    for index, label in index_to_label_dict.items():
        label_to_index[label] = index

    return label_to_index


def apply_labels(
    basic_blocks: list[BasicBlock], index_to_label: IndexToLabelDict
) -> list[BasicBlock]:
    """Returns new basic blocks with labels"""
    basic_blocks_with_labels = copy.deepcopy(basic_blocks)
    for i, _ in enumerate(basic_blocks):
        if not has_label(basic_blocks_with_labels[i]):
            label = Label({"label": index_to_label[i]})
            basic_blocks_with_labels[i].insert(0, label)

    return basic_blocks_with_labels


@click.command()
def main():
    prog: Program = json.load(sys.stdin)
    bb_program: BasicBlockProgram = basic_block_program_from_program(prog)

    for func in bb_program["functions"]:
        cfg = control_flow_graph_from_instructions(func["instrs"])
        index_to_label = index_to_label_dict_get(func, cfg)
        func["instrs"] = apply_labels(func["instrs"], index_to_label)

    print(json.dumps(program_from_basic_block_program(bb_program)))


if __name__ == "__main__":
    main()
