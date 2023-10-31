"""Calls brili from Python"""
import json
import re
import subprocess
import sys
from typing import NamedTuple

import click

from bril.typing_bril import Program

BRILI_COMMAND = "brili"
BRILI_PROFILE_FLAG = "-p"
BRILI_PROFILE_REGEX = re.compile(r"^total_dyn_inst: (\d+)$")
BRILI_ERROR_REGEX = re.compile(r"^error: (.*)$")

class BriliOutput(NamedTuple):
    """What brili would output"""
    returncode: int
    stdout: str
    stderr: str
    error: str | None
    total_dyn_inst: int | None


def brili(program: Program, arguments: list[str], profile: bool = False) -> BriliOutput:
    """Interface to brili"""
    command = [BRILI_COMMAND]
    if profile:
        command.append(BRILI_PROFILE_FLAG)

    command.extend(arguments)
    result = subprocess.run( # pylint: disable=subprocess-run-check
        command, input=json.dumps(program).encode("utf-8"), capture_output=True
    )

    stderr = result.stderr.decode("utf-8")

    error: str | None = None
    total_dyn_inst: int | None = None

    for line in stderr.splitlines():
        match_ = BRILI_ERROR_REGEX.match(line)
        if match_ is not None:
            error = match_.groups()[0]
            continue

        match_ = BRILI_PROFILE_REGEX.match(line)
        if match_ is not None:
            total_dyn_inst = int(match_.groups()[0])

    return BriliOutput(
        returncode=result.returncode,
        stdout=result.stdout.decode("utf-8"),
        stderr=stderr,
        error=error,
        total_dyn_inst=total_dyn_inst,
    )


@click.command()
@click.option(
    "-p",
    "--profile",
    is_flag=True,
    default=False,
    type=bool,
    help="Profile total number of dynamic instructions",
)
@click.argument("arguments", nargs=-1)
def main(profile: bool, arguments: list[str]):
    program: Program = json.load(sys.stdin)
    print(brili(program, arguments, profile=profile))


if __name__ == "__main__":
    main()