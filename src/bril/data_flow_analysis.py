"""Data Flow Analysis"""

import json
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator, Generic, TypeAlias, TypeVar, cast

import click

from bril.basic_blocks import (
    BasicBlock,
    BasicBlockFunction,
    BasicBlockProgram,
    basic_block_program_from_program,
)
from bril.bril_extract import label_get
from bril.cfg import ControlFlowGraph, control_flow_graph_from_instructions
from bril.typing_bril import Effect, Program, Value

Domain = TypeVar("Domain")


class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2


def unique_label_name_generator(used_labels: set[str]) -> Generator[str, None, None]:
    """Generates new labels"""
    i = 0
    while True:
        candidate = f"b{i}"
        if candidate not in used_labels:
            yield candidate
            used_labels.add(candidate)


def index_to_label_dict_get(
    basic_blocks: list[BasicBlock], cfg: ControlFlowGraph
) -> dict[int, str]:
    """
    Assigns basic block indices to labels.
    Creates new labels if a block does not have a label.
    Does not modify the code.
    """
    index_to_label: dict[int, str] = {}
    no_label_indices = []
    for i, basic_block in enumerate(basic_blocks):
        label = label_get(basic_block)
        if label is not None:
            index_to_label[i] = label
        else:
            no_label_indices.append(i)

    label_name_generator = unique_label_name_generator(set(index_to_label.values()))
    for i in no_label_indices:
        index_to_label[i] = next(label_name_generator)

    index_to_label[cfg.entry] = "ENTRY"
    index_to_label[cfg.exit] = "EXIT"

    return index_to_label


class DataFlowAnalysis(ABC, Generic[Domain]):
    @property
    @abstractmethod
    def direction(self) -> Direction:
        """Data flow direction"""

    @abstractmethod
    def initial_in_out(
        self, func: BasicBlockFunction, cfg: ControlFlowGraph
    ) -> tuple[dict[int, Domain], dict[int, Domain]]:
        """Initial in and out dictionaries for the analysis algorithm"""

    @abstractmethod
    def merge(self, *domains: Domain) -> Domain:
        """Merge the domains for worklist algorithm"""

    @abstractmethod
    def transfer(
        self, basic_block: BasicBlock, basic_block_index: int, source: Domain
    ) -> Domain:
        """Transfer the source set to another set using basic_block"""

    @staticmethod
    @abstractmethod
    def sprint_domain(domain: Domain, index_to_label: dict[int, str]) -> str:
        """Returns a pretty string of an element in the domain"""


def analyze_data_flow(
    data_flow_analysis: DataFlowAnalysis[Domain],
    basic_block_function: BasicBlockFunction,
    cfg: ControlFlowGraph,
) -> tuple[dict[int, Domain], dict[int, Domain]]:
    """
    Given a data flow analysis object, basic blocks, and their control flow graph,
    analyze their data flows
    """
    in_, out = data_flow_analysis.initial_in_out(basic_block_function, cfg)

    if data_flow_analysis.direction == Direction.FORWARD:
        dict1 = in_
        dict2 = out
        endpoint1_get = cfg.predecessors
        endpoint2_get = cfg.successors
    else:
        dict1 = out
        dict2 = in_
        endpoint1_get = cfg.successors
        endpoint2_get = cfg.predecessors

    worklist = set(range(len(basic_block_function["instrs"])))
    while len(worklist) > 0:
        b_index = worklist.pop()
        dict1[b_index] = data_flow_analysis.merge(
            *(dict2[i] for i in endpoint1_get(b_index))
        )

        basic_block: BasicBlock
        if 0 <= b_index < len(basic_block_function["instrs"]):
            basic_block = basic_block_function["instrs"][b_index]
        else:
            basic_block = []

        new_dict2_set = data_flow_analysis.transfer(
            basic_block, b_index, dict1[b_index]
        )

        if dict2[b_index] != new_dict2_set:
            worklist.update(endpoint2_get(b_index))

        dict2[b_index] = new_dict2_set

    return dict(in_), dict(out)


DefinitionIdentifier: TypeAlias = tuple[str, int]


class ReachingDefinitions(DataFlowAnalysis[set[DefinitionIdentifier]]):
    """Bookkeep what definitions are available"""

    @property
    def direction(self) -> Direction:
        """Data flow direction"""
        return Direction.FORWARD

    def initial_in_out(
        self, func: BasicBlockFunction, cfg: ControlFlowGraph
    ) -> tuple[
        dict[int, set[DefinitionIdentifier]], dict[int, set[DefinitionIdentifier]]
    ]:
        """Initial in and out dictionaries for the analysis algorithm"""
        in_: dict[int, set[DefinitionIdentifier]] = {
            i: set() for i, _ in enumerate(func["instrs"])
        }
        out: dict[int, set[DefinitionIdentifier]] = {
            i: set() for i, _ in enumerate(func["instrs"])
        }
        in_[cfg.entry] = set()
        out[cfg.entry] = set()
        in_[cfg.exit] = set()
        out[cfg.exit] = set()

        if "args" in func:
            out[cfg.entry].update((arg["name"], cfg.entry) for arg in func["args"])

        return in_, out

    def merge(self, *domains: set[DefinitionIdentifier]) -> set[DefinitionIdentifier]:
        """Merge the domains for worklist algorithm"""
        return set().union(*domains)

    def transfer(
        self,
        basic_block: BasicBlock,
        basic_block_index: int,
        source: set[DefinitionIdentifier],
    ) -> set[DefinitionIdentifier]:
        """Transfer the source set to another set using basic_block"""
        dest = source.copy()
        for instruction in basic_block:
            if "dest" in instruction:
                value = cast(Value, instruction)
                for value_dest, bb_index in source:
                    if value_dest == value["dest"]:
                        dest.remove((value_dest, bb_index))

                dest.add((value["dest"], basic_block_index))
        return dest

    @staticmethod
    def sprint_domain(
        domain: set[DefinitionIdentifier], index_to_label: dict[int, str]
    ) -> str:
        """Returns a pretty string of an element in the domain"""
        if len(domain) <= 0:
            return "∅"

        def sprintf_element(element: DefinitionIdentifier):
            variable_name, basic_block_index = element
            return f"{index_to_label[basic_block_index]}.{variable_name}"

        return ", ".join([sprintf_element(element) for element in sorted(list(domain))])


class LiveVariables(DataFlowAnalysis[set[str]]):
    """Bookkeep what variables are live"""

    @property
    def direction(self) -> Direction:
        """Data flow direction"""
        return Direction.BACKWARD

    def initial_in_out(
        self,
        func: BasicBlockFunction,
        cfg: ControlFlowGraph,
    ) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
        """Initial in and out dictionaries for the analysis algorithm"""
        in_: dict[int, set[str]] = {i: set() for i, _ in enumerate(func["instrs"])}
        out: dict[int, set[str]] = {i: set() for i, _ in enumerate(func["instrs"])}
        in_[cfg.entry] = set()
        out[cfg.entry] = set()
        in_[cfg.exit] = set()
        out[cfg.exit] = set()

        return in_, out

    def merge(self, *domains: set[str]) -> set[str]:
        """Merge the domains for worklist algorithm"""
        return set().union(*domains)

    def transfer(
        self,
        basic_block: BasicBlock,
        basic_block_index: int,
        source: set[str],
    ) -> set[str]:
        """Transfer the source set to another set using basic_block"""
        dest = source.copy()
        for instruction in reversed(basic_block):
            if "dest" in instruction:
                # kill definitions
                value = cast(Value, instruction)
                dest.discard(value["dest"])
            if "args" in instruction:
                # add arguments
                effect = cast(Effect, instruction)
                dest.update(effect["args"])
        return dest

    @staticmethod
    def sprint_domain(domain: set[str], index_to_label: dict[int, str]) -> str:
        """Returns a pretty string of an element in the domain"""
        if len(domain) <= 0:
            return "∅"

        return ", ".join(sorted(list(domain)))


DATA_FLOW_ANALYSIS_MAP: dict[str, DataFlowAnalysis] = {
    "defined": ReachingDefinitions(),
    "live": LiveVariables(),
}


@click.command()
@click.argument(
    "data-flow-analysis-type",
    type=click.Choice(DATA_FLOW_ANALYSIS_MAP.keys(), case_sensitive=False),
)
def main(data_flow_analysis_type):
    data_flow_analysis = DATA_FLOW_ANALYSIS_MAP[data_flow_analysis_type]
    prog: Program = json.load(sys.stdin)
    bb_program: BasicBlockProgram = basic_block_program_from_program(prog)

    for func in bb_program["functions"]:
        cfg = control_flow_graph_from_instructions(func["instrs"])
        index_to_label = index_to_label_dict_get(func["instrs"], cfg)
        in_, out = analyze_data_flow(data_flow_analysis, func, cfg)
        print(f'{func["name"]}:')
        for i, basic_block in enumerate(func["instrs"]):
            label = label_get(basic_block)
            if label is not None:
                block_name = label
            else:
                block_name = f"b{i}"
            print(f"  {block_name}:")
            in_string = data_flow_analysis.sprint_domain(in_[i], index_to_label)
            print(f"    in:  {in_string}")
            out_string = data_flow_analysis.sprint_domain(out[i], index_to_label)
            print(f"    out: {out_string}")


if __name__ == "__main__":
    main()
