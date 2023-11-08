"""
Constants used in Bril tools
"""
from typing import Literal

from bril.typing_bril import BrilType, Operation

ENTRY_FUNCTION_NAME = "main"

TERMINATOR_OPERATORS: list[Operation] = ["jmp", "br", "ret"]
OPERATIONS_WITH_SIDE_EFFECTS: list[Operation] = ["call", "print", "alloc"]
COMMUTATIVE_OPERATIONS: list[Operation] = [
    "add",
    "mul",
    "eq",
    "and",
    "or",
    "fadd",
    "fmul",
    "feq",
    "ceq",
]

GenericType = Literal["generic"]

OPERATORS: tuple[
    tuple[Operation, tuple[BrilType | GenericType, ...], BrilType | GenericType | None]
] = (
    ("const", ("generic",), "generic"),
    # core
    ("add", ("int", "int"), "int"),
    ("mul", ("int", "int"), "int"),
    ("sub", ("int", "int"), "int"),
    ("div", ("int", "int"), "int"),
    ("eq", ("int", "int"), "int"),
    ("lt", ("int", "int"), "bool"),
    ("gt", ("int", "int"), "bool"),
    ("le", ("int", "int"), "bool"),
    ("ge", ("int", "int"), "bool"),
    ("not", ("bool"), "bool"),
    ("and", ("bool", "bool"), "bool"),
    ("or", ("bool", "bool"), "bool"),
    ("jmp", (), None),
    ("br", ("bool"), None),
    ("call", (), None),  # TODO Need to generalize this
    ("ret", ("generic",), None),
    ("id", ("generic",), "generic"),
    ("print", ("generic"), None),  # TODO Need to do variadic
    ("nop", (), None),
    # memory
    ("free", ({"ptr": "generic"},), None),
    ("alloc", ("int",), {"ptr": "generic"}),
    ("store", ({"ptr": "generic"}, "generic"), None),
    ("load", ({"ptr": "generic"},), "generic"),
    ("ptradd", ({"ptr": "generic"}, "int"), {"ptr": "generic"}),
    # float
    ("fadd", ("float", "float"), "float"),
    ("fmul", ("float", "float"), "float"),
    ("fsub", ("float", "float"), "float"),
    ("fdiv", ("float", "float"), "float"),
    ("feq", ("float", "float"), "float"),
    ("flt", ("float", "float"), "bool"),
    ("fle", ("float", "float"), "bool"),
    ("fgt", ("float", "float"), "bool"),
    ("fge", ("float", "float"), "bool"),
    # SSA
    ("phi", ("generic",), "generic"),  # TODO Need to do variadic
    # speculative execution
    ("speculate", (), None),
    ("commit", (), None),
    ("guard", (), None),
    # character
    ("ceq", ("char", "char"), "bool"),
    ("clt", ("char", "char"), "bool"),
    ("cle", ("char", "char"), "bool"),
    ("cgt", ("char", "char"), "bool"),
    ("cge", ("char", "char"), "bool"),
    ("char2int", ("char",), "int"),
    ("int2char", ("int",), "char"),
)
