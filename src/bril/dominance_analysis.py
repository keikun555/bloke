"""
Utilities to find dominators and the likes
"""

import copy
import json
import sys
from collections import defaultdict
from typing import Callable, TypeAlias

import click

from .basic_blocks import BasicBlockProgram, basic_block_program_from_program
from .bril_labeler import index_to_label_dict_get
from .cfg import (ControlFlowGraph, all_paths,
                  control_flow_graph_from_instructions, reverse_cfg)
from .typing_bril import Program

DominanceAnalysis: TypeAlias = dict[int, set[int]]


def dominators_indices_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index to sets of dominators of the block index"""
    doms: DominanceAnalysis = defaultdict(set)
    for vertex in cfg:
        doms[vertex] = set(cfg.keys())
    doms[cfg.entry] = {cfg.entry}

    all_indices = set([cfg.entry, cfg.exit] + list(cfg.keys()))

    changing = True
    while changing:
        changing = False

        for vertex in cfg:
            if vertex == cfg.entry:
                continue

            intersection = all_indices.copy()
            for predecessor in cfg.predecessors(vertex):
                intersection &= doms[predecessor]

            new_dominator_indices = {vertex} | intersection

            if doms[vertex] != new_dominator_indices:
                changing = True

            doms[vertex] = new_dominator_indices

    return doms


def naive_dominators_indices_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index to sets of dominators of the block index"""
    all_indices = set([cfg.entry, cfg.exit] + list(cfg.keys()))
    doms: DominanceAnalysis = {i: all_indices.copy() for i in cfg}
    for i in cfg:
        for path in all_paths(cfg, cfg.entry, i):
            doms[i] &= set(path)

    return doms


def strict_dominators_indices_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index
    to sets of strict dominators of the block index"""
    strict_dominators: DominanceAnalysis = {}

    dominators = dominators_indices_get(cfg)
    for i in cfg:
        strict_dominators[i] = dominators[i] - {i}

    return strict_dominators


def naive_strict_dominators_indices_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index
    to sets of strict dominators of the block index"""
    all_indices = set([cfg.entry, cfg.exit] + list(cfg.keys()))
    sdoms: DominanceAnalysis = {i: all_indices.copy() - {i} for i in cfg}
    for i in cfg:
        for path in all_paths(cfg, cfg.entry, i):
            for _, vertex in enumerate(path):
                sdoms[vertex] &= set(path) - {vertex}

    return sdoms


def immediate_dominator_index_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index
    to sets of immediate dominators of the block index"""
    idom: DominanceAnalysis = {}

    strict_dominators = strict_dominators_indices_get(cfg)

    for i in cfg:
        idom[i] = strict_dominators[i].copy()
        for node in strict_dominators[i]:
            idom[i] -= strict_dominators[node]

        if len(idom[i]) > 1:
            # immediate dominator is not well-defined
            # set to nil
            idom[i].clear()

    return idom


def naive_immediate_dominator_index_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index
    to sets of immediate dominators of the block index"""
    strict_dominators = naive_strict_dominators_indices_get(cfg)

    idom: DominanceAnalysis = copy.deepcopy(strict_dominators)
    for i in cfg:
        for path in all_paths(cfg, cfg.entry, i):
            for _, vertex in enumerate(path):
                idom[vertex] &= set(path)
                idom[vertex] -= {vertex}  # strict domination

    for i in cfg:
        for node in strict_dominators[i]:
            # idom[i] does not strictly dominate
            # any other node that strictly dominates vertex
            idom[i] -= strict_dominators[node]

        if len(idom[i]) > 1:
            # immediate dominator is not well-defined
            # set to nil
            idom[i].clear()

    return idom


def index_dominator_tree_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dominator tree in form of a dictionary of block index
    to its immediate dominator index"""
    dominator_tree: DominanceAnalysis = {i: set() for i in cfg}
    immediate_dominators = immediate_dominator_index_get(cfg)

    for i, idoms in immediate_dominators.items():
        if len(idoms) == 1:
            dominator_tree[idoms.pop()].add(i)

    return dominator_tree


def naive_index_dominator_tree_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dominator tree in form of a dictionary of block index
    to its immediate dominator index"""
    dominator_tree: DominanceAnalysis = {i: set() for i in cfg}
    immediate_dominators = naive_immediate_dominator_index_get(cfg)

    for i, idoms in immediate_dominators.items():
        assert (
            len(idoms) <= 1
        ), f"block index {i} has more than one immediate dominator: {idoms}"

        if len(idoms) == 1:
            dominator_tree[idoms.pop()].add(i)

    return dominator_tree


def dominance_frontier_indices_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index
    to sets of dominance frontiers of the block index"""
    frontier: DominanceAnalysis = {}

    dominators = dominators_indices_get(cfg)
    strict_dominators = strict_dominators_indices_get(cfg)
    for i in cfg:
        frontier[i] = set()
        for j in cfg:
            predecessors = set().union(*[dominators[k] for k in cfg.predecessors(j)])
            if i in predecessors and i not in strict_dominators[j]:
                # i dominates a predecessor of frontier
                # i does not strictly dominate frontier
                frontier[i].add(j)

    return frontier


def naive_dominance_frontier_indices_get(cfg: ControlFlowGraph) -> DominanceAnalysis:
    """Given CFG, return dictionary mapping block index
    to sets of dominance frontiers of the block index"""
    frontier: DominanceAnalysis = {}

    dominators = naive_dominators_indices_get(cfg)
    strict_dominators = naive_strict_dominators_indices_get(cfg)
    for i in cfg:
        frontier[i] = set()
        for j in cfg:
            predecessors = set().union(*[dominators[k] for k in cfg.predecessors(j)])
            if i in predecessors and i not in strict_dominators[j]:
                # i dominates a predecessor of frontier
                # i does not strictly dominate frontier
                frontier[i].add(j)

    return frontier


DOMINANCE_ANALYSIS_MAP: dict[str, Callable[[ControlFlowGraph], DominanceAnalysis]] = {
    # dominance
    "dominance": dominators_indices_get,
    "naive-dominance": naive_dominators_indices_get,
    # strict dominance
    "strict-dominance": strict_dominators_indices_get,
    "naive-strict-dominance": naive_strict_dominators_indices_get,
    # immediate dominance
    "immediate-dominance": immediate_dominator_index_get,
    "naive-immediate-dominance": naive_immediate_dominator_index_get,
    # dominator tree
    "dominator-tree": index_dominator_tree_get,
    "naive-dominator-tree": naive_index_dominator_tree_get,
    # dominance frontier
    "dominance-frontier": dominance_frontier_indices_get,
    "naive-dominance-frontier": naive_dominance_frontier_indices_get,
}


@click.command()
@click.argument(
    "dominance-analysis-type",
    type=click.Choice(DOMINANCE_ANALYSIS_MAP.keys(), case_sensitive=False),
)
@click.option(
    "-p",
    "--post",
    is_flag=True,
    default=False,
    type=bool,
    help="Same analysis but with post-dominance instead of dominance",
)
def main(dominance_analysis_type: str, post: bool):
    analysis = DOMINANCE_ANALYSIS_MAP[dominance_analysis_type]
    prog: Program = json.load(sys.stdin)
    bb_program: BasicBlockProgram = basic_block_program_from_program(prog)
    analysis_result: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for func in bb_program["functions"]:
        cfg = control_flow_graph_from_instructions(func["instrs"])
        if post:
            # post-dominance analysis is the same as dominance analysis
            # just with cfg edges reversed
            cfg = reverse_cfg(cfg)

        hallucinated_blocks = (cfg.entry, cfg.exit)

        index_to_label = index_to_label_dict_get(func, cfg)

        dominance_analysis = analysis(cfg)
        for i, _ in enumerate(func["instrs"]):
            dominators_list = sorted(
                [
                    j
                    for j in dominance_analysis.get(i, set())
                    if j not in hallucinated_blocks
                ]
            )
            analysis_result[func["name"]][index_to_label[i]] = [
                index_to_label[j] for j in dominators_list
            ]

    print(json.dumps(analysis_result))


if __name__ == "__main__":
    main()
