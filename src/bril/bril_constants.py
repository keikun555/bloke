"""
Constants used in Bril tools
"""

from bril.typing_bril import Operation

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
