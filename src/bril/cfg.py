"""
Generates a Control Flow Graph from a BasicBlockFunction
"""
import json
import sys
import copy


from typing import cast, Generator

from typing_bril import (
    Program,
    InstructionBase,
    Effect,
)
from bril_constants import TERMINATOR_OPERATORS

from basic_blocks import (
    basic_block_program_from_program,
    BasicBlock,
)
from bril_extract import label_get


class ControlFlowGraph(dict[int, set[int]]):
    """
    Key: index of basic block in function
    Value: set of indices of basic blocks that map from the basic block
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # entry and exits are sets of indices of entry blocks and exit blocks, respectively
        self.entry = -1
        self.exit = (
            -2
        )  # this will be overwritten when calling control_flow_graph_from_instructions

    def reachable(self, start_index: int, dest_index: int) -> bool:
        """Tells us whether a block is reachable from the entry"""
        visited: set[int] = set()

        def helper(i: int):
            visited.add(i)
            if i == dest_index:
                return True
            for j in self[i]:
                if j not in visited and helper(j):
                    return True
            return False

        return helper(start_index)

    @property
    def edges(self) -> set[tuple[int, int]]:
        return {(i, j) for i in self for j in self[i]}

    def predecessors(self, block_index: int) -> tuple[int, ...]:
        """Given block index, return predecessor indices of block"""
        return tuple(sorted([i for i in self if block_index in self[i]]))

    def successors(self, block_index: int) -> tuple[int, ...]:
        """Given block index, return successor indices of block"""
        return tuple(sorted(list(self[block_index])))


def control_flow_graph_from_instructions(
    basic_blocks: list[BasicBlock],
) -> ControlFlowGraph:
    """Given list of basic blocks generate a control flow graph"""
    cfg = ControlFlowGraph({i: set() for i, _ in enumerate(basic_blocks)})

    # Treat -1 as entry block and len(basic_blocks) as exit block
    cfg.entry = -1
    cfg.exit = len(basic_blocks)

    cfg[cfg.entry] = set()
    cfg[cfg.exit] = set()

    if len(basic_blocks) <= 0:
        cfg[cfg.entry].add(cfg.exit)
        return cfg

    cfg[cfg.entry].add(0)

    labels_to_index: dict[str, int] = {}
    for i, basic_block in enumerate(basic_blocks):
        label = label_get(basic_block)
        if label is not None:
            labels_to_index[label] = i

    for i, basic_block in enumerate(basic_blocks):
        if len(basic_block) <= 0:
            cfg[i].add(i + 1)
            continue
        last_instruction = basic_block[-1]
        if "op" in last_instruction:
            instruction = cast(InstructionBase, last_instruction)

            if instruction["op"] in TERMINATOR_OPERATORS:
                if instruction["op"] == "ret":
                    cfg[i].add(cfg.exit)
                    continue

                terminator = cast(Effect, instruction)
                if "labels" in terminator:
                    cfg[i].update(
                        labels_to_index[label] for label in terminator["labels"]
                    )

            else:
                cfg[i].add(i + 1)
        else:
            cfg[i].add(i + 1)

    return cfg


class ReversedControlFlowGraph(ControlFlowGraph):
    """This is a CFG but instead of mapping to successors, maps to predecessors"""


def reverse_cfg(cfg: ControlFlowGraph) -> ReversedControlFlowGraph:
    """Reverse a CFG"""
    reversed_cfg = ReversedControlFlowGraph(copy.deepcopy(cfg))

    reversed_cfg.entry = cfg.exit
    reversed_cfg.exit = cfg.entry

    for i in cfg:
        reversed_cfg[i] = set(cfg.predecessors(i))

    return reversed_cfg


def all_paths(
    cfg: ControlFlowGraph, start_index: int, end_index: int
) -> Generator[tuple[int, ...], None, None]:
    """Iterates through all paths in CFG from start block to end block
    Algorithm inspired from https://www.geeksforgeeks.org/find-paths-given-source-destination/"""
    visited: set[int] = {cfg.entry}

    def helper(vertex: int, path: list[int]) -> Generator[tuple[int, ...], None, None]:
        visited.add(vertex)
        path.append(vertex)
        if vertex == end_index:
            yield tuple(path)
        else:
            for successor in cfg.successors(vertex):
                if successor not in visited:
                    yield from helper(successor, path)

        path.pop()
        visited.discard(vertex)

    yield from helper(start_index, [])


def unreachable_blocks(cfg: ControlFlowGraph) -> set[int]:
    """Return blocks unreachable from entry"""
    visited: set[int] = set()

    def helper(vertex: int) -> None:
        visited.add(vertex)

        for successor in cfg.successors(vertex):
            if successor not in visited:
                helper(successor)

    helper(cfg.entry)

    return set(cfg.keys()) - visited


def is_cyclic(cfg: ControlFlowGraph) -> bool:
    """Returns True iff given cfg is not cyclic"""
    # https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
    visited: set[int] = set()
    stack: set[int] = set()

    def helper(vertex: int) -> bool:
        visited.add(vertex)
        stack.add(vertex)

        # Recur for all neighbours
        # if any neighbour is visited and in
        # stack then graph is cyclic
        for successor in cfg.successors(vertex):
            if successor not in visited:
                if helper(successor):
                    return True
            elif successor in stack:
                return True

        # The node needs to be popped from
        # recursion stack before function ends
        stack.discard(vertex)
        return False

    return helper(cfg.entry)


def main():
    prog: Program = json.load(sys.stdin)
    basic_block_program = basic_block_program_from_program(prog)
    for func in basic_block_program["functions"]:
        print(
            f'{func["name"]}:\t{control_flow_graph_from_instructions(func["instrs"])}'
        )


if __name__ == "__main__":
    main()
