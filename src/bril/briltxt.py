"""briltxt but with prints"""
import sys
from io import StringIO
from typing import TypeAlias

import briltxt

from bril.typing_bril import Program

BrilText: TypeAlias = str


def prints_prog(program: Program) -> BrilText:
    """Takes in a Bril program (dictionary), prints out Bril in human-readable text form"""
    string_io = StringIO()
    sys.stdout = string_io
    briltxt.print_prog(program)
    sys.stdout = sys.__stdout__
    return string_io.getvalue().strip()
