"""
Generates basic block structures from brili json dictionaries
"""
import json
import sys
import itertools

from typing import Generator, TypeAlias
from typing_extensions import TypedDict

from bril.typing_bril import Function, FunctionBase, Instruction, Program
from bril.bril_constants import TERMINATOR_OPERATORS


BasicBlock: TypeAlias = list[Instruction]


class BasicBlockFunction(FunctionBase):
    instrs: list[BasicBlock]


class BasicBlockProgram(TypedDict):
    functions: list[BasicBlockFunction]


def basic_block_generator(func: Function) -> Generator[BasicBlock, None, None]:
    """given a function generates the basic block decomposition of the function instructions"""
    cur_block = []
    for instr in func["instrs"]:
        if instr.get("op") in TERMINATOR_OPERATORS:
            cur_block.append(instr)
            yield cur_block
            cur_block = []
        elif "label" in instr:
            if len(cur_block) > 0:
                yield cur_block
            cur_block = [instr]
        else:
            cur_block.append(instr)
    if len(cur_block) > 0:
        yield cur_block


def basic_block_program_from_program(prog: Program) -> BasicBlockProgram:
    """From a JSON Bril program in dictionary form create a BasicBlockProgram"""
    bb_prog: BasicBlockProgram = {"functions": []}
    for func in prog["functions"]:
        basic_blocks = list(basic_block_generator(func))
        bb_func: BasicBlockFunction = {
            "name": func["name"],
            "instrs": basic_blocks,
        }
        if "args" in func:
            bb_func["args"] = func["args"]
        if "type" in func:
            bb_func["type"] = func["type"]
        bb_prog["functions"].append(bb_func)

    return bb_prog


def program_from_basic_block_program(bb_prog: BasicBlockProgram) -> Program:
    """From a BasicBlockProgram create a Bril program"""
    prog: Program = {"functions": []}
    for bb_func in bb_prog["functions"]:
        instructions = list(itertools.chain.from_iterable(bb_func["instrs"]))
        func: Function = {
            "name": bb_func["name"],
            "instrs": instructions,
        }
        if "args" in bb_func:
            func["args"] = bb_func["args"]
        if "type" in bb_func:
            func["type"] = bb_func["type"]
        prog["functions"].append(func)

    return prog


def main():
    prog: Program = json.load(sys.stdin)
    basic_block_program = basic_block_program_from_program(prog)
    print(json.dumps(basic_block_program))
    assert json.dumps(
        program_from_basic_block_program(basic_block_program), sort_keys=True
    ) == json.dumps(prog, sort_keys=True)


if __name__ == "__main__":
    main()
