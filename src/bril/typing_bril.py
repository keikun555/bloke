"""Defines types for bril"""
from typing import Literal, TypeAlias, TypedDict

from typing_extensions import NotRequired

Operation: TypeAlias = Literal[
    "const",
    # core
    "add",
    "mul",
    "sub",
    "div",
    "eq",
    "lt",
    "gt",
    "le",
    "ge",
    "not",
    "and",
    "or",
    "jmp",
    "br",
    "call",
    "ret",
    "id",
    "print",
    "nop",
    # memory
    "free",
    "alloc",
    "store",
    "load",
    "ptradd",
    # float
    "fadd",
    "fmul",
    "fsub",
    "fdiv",
    "feq",
    "flt",
    "fle",
    "fgt",
    "fge",
    # SSA
    "phi",
    # speculative execution
    "speculate",
    "commit",
    "guard",
    # character
    "ceq",
    "clt",
    "cle",
    "cgt",
    "cge",
    "char2int",
    "int2char",
]


class Position(TypedDict):
    row: int
    col: int


class Syntax(TypedDict):
    pos: NotRequired[Position]
    pos_end: NotRequired[Position]
    src: NotRequired[str]


class Label(TypedDict):
    label: str


BrilType: TypeAlias = (
    Literal["int", "float", "bool", "char"] | dict[Literal["ptr"], "BrilType"]
)


class InstructionBase(TypedDict):
    op: Operation


class Variable(str):
    ...


class ValueInstructionBase(InstructionBase):
    dest: Variable
    type: BrilType


PrimitiveType: TypeAlias = int | bool | float


class Constant(ValueInstructionBase):
    value: PrimitiveType


class NonConstantInstructionMixin(TypedDict):
    args: NotRequired[list[Variable]]
    funcs: NotRequired[list[str]]
    labels: NotRequired[list[str]]


class Value(ValueInstructionBase, NonConstantInstructionMixin):
    ...


class Effect(InstructionBase, NonConstantInstructionMixin):
    ...


Instruction: TypeAlias = Constant | Value | Effect | Label


class Argument(TypedDict):
    name: Variable
    type: BrilType


class FunctionBase(TypedDict):
    name: str
    args: NotRequired[list[Argument]]
    type: NotRequired[str]


class Function(FunctionBase):
    instrs: list[Instruction]


class Program(TypedDict):
    functions: list[Function]
