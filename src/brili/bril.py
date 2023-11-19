from typing import Literal, TypedDict

from typing_extensions import NotRequired

# A variable name.
Ident = str

# Primitive types.
PrimType = Literal["int", "bool", "float", "char"]

# Value types.
Type = PrimType | dict[Literal["ptr"], "Type"]
ParamType = dict[Literal["ptr"], Type]

# An (always optional) source code position.
Position = dict[str, int]

# Common fields in any operation.
class Op(TypedDict):
    args: NotRequired[list[Ident]]
    funcs: NotRequired[list[Ident]]
    labels: NotRequired[list[Ident]]
    pos: NotRequired[Position]


# The valid opcodes for value-producing instructions.
ValueOpCode = Literal[
    "add",
    "mul",
    "sub",
    "div",
    "id",
    "nop",
    "eq",
    "lt",
    "gt",
    "ge",
    "le",
    "not",
    "and",
    "or",
    "call",
    "load",
    "ptradd",
    "alloc",
    "fadd",
    "fmul",
    "fsub",
    "fdiv",
    "feq",
    "flt",
    "fle",
    "fgt",
    "fge",
    "ceq",
    "clt",
    "cle",
    "cgt",
    "cge",
    "char2int",
    "int2char",
    "phi",
]

# The valid opcodes for effecting operations.
EffectOpCode = Literal[
    "br",
    "jmp",
    "print",
    "ret",
    "call",
    "store",
    "free",
    "speculate",
    "guard",
    "commit",
]


# An instruction that does not produce any result.
class EffectOperation(Op):
    op: EffectOpCode


# An operation that produces a value and places its result in the
# destination variable.
class ValueOperation(Op):
    op: ValueOpCode
    dest: Ident
    type: Type


# The type of Bril values that may appear in constants.
Value = int | bool | str

# An instruction that places a literal value into a variable.
class Constant(TypedDict):
    op: Literal["const"]
    value: Value
    dest: Ident
    type: Type
    pos: NotRequired[Position]


# Operations take arguments, which come from previously-assigned identifiers.
Operation = EffectOperation | ValueOperation

# Instructions can be operations (which have arguments) or constants (which
# don't). Both produce a value in a destination variable.
Instruction = Operation | Constant

# Both constants and value operations produce results.
ValueInstruction = Constant | ValueOperation

# All valid operation opcodes.
OpCode = ValueOpCode | EffectOpCode

# Jump labels just mark a position with a name.
class Label(TypedDict):
    label: Ident
    pos: NotRequired[Position]


# An argument has a name and a type.
class Argument(TypedDict):
    name: Ident
    type: Type


# A function consists of a sequence of instructions.
class Function(TypedDict):
    name: Ident
    args: NotRequired[list[Argument]]
    instrs: NotRequired[list[Instruction | Label]]
    type: NotRequired[Type]
    pos: NotRequired[Position]


# A program consists of a set of functions, one of which must be named "main".
class Program(TypedDict):
    functions: list[Function]
