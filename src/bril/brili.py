"""Calls brili from Python"""
import abc
import json
import re
import subprocess
import sys
from io import StringIO
from typing import Iterable, NamedTuple, cast

import click

import brilirs
from bril.briltxt import prints_prog
from bril.typing_bril import PrimitiveType, Program

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


class Brili(abc.ABC):
    """Abstract interface for the brili interpreter"""

    @abc.abstractmethod
    def interpret(
        self,
        program: Program,
        arguments: Iterable[PrimitiveType],
        profile: bool = False,
    ) -> BriliOutput:
        """Interpret the program, return BriliOutput"""
        ...

    @abc.abstractmethod
    def interpret_json(
        self, program: dict, arguments: Iterable[str], profile: bool = False
    ) -> BriliOutput:
        """Interpret string program and string input"""
        ...


def python_to_brili_argument(argument: PrimitiveType) -> str:
    """Convert Python's Bril primitive type to brili argument"""
    if isinstance(argument, (int, float)):
        return str(argument)
    if isinstance(argument, bool):
        return str(argument).lower()

    raise ValueError(f"python_to_brili_argument, unsupported: {argument}")


class SubprocessBrili(Brili):
    @staticmethod
    def interpret(
        program: Program,
        arguments: Iterable[PrimitiveType],
        profile: bool = False,
    ) -> BriliOutput:
        """Interface to brili, supports Python's Bril primitive type"""
        string_arguments = (python_to_brili_argument(arg) for arg in arguments)
        return SubprocessBrili.interpret_json(
            cast(dict, program), string_arguments, profile=profile
        )

    @staticmethod
    def interpret_json(
        program: dict, arguments: Iterable[str], profile: bool = False
    ) -> BriliOutput:
        """Interface to brili, takes in raw strings for arguments"""
        program_string = json.dumps(program).encode("utf-8")
        command = [BRILI_COMMAND]
        if profile:
            command.append(BRILI_PROFILE_FLAG)

        command.extend(arguments)
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            command,
            input=program_string,
            capture_output=True,
            shell=False,
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


class Brilirs(Brili):
    def interpret(
        self,
        program: Program,
        arguments: Iterable[PrimitiveType],
        profile: bool = False,
    ) -> BriliOutput:
        """Interface to brili, supports Python's Bril primitive type"""
        string_arguments = [python_to_brili_argument(arg) for arg in arguments]

        return Brilirs.interpret_string(json.dumps(program), string_arguments, profile)

    @staticmethod
    def interpret_json(program: dict, arguments: Iterable[str], profile: bool = False):
        program_string = prints_prog(cast(Program, program))

        return Brilirs.interpret_string(program_string, list(arguments), profile)

    @staticmethod
    def interpret_string(program: str, arguments: list[str], profile: bool = False):
        stdout_list: list[str] = []
        stderr_list: list[str] = []
        error: str | None
        result: int | None = None

        try:
            result = brilirs.run_program(
                program,
                stdout_list,
                arguments,
                profile,
                stderr_list,
            )
            error = None
        except Exception as e:
            error = str(e)
            result = 2

        if result is None:
            returncode = 0
        else:
            returncode = result

        stdout = "".join(stdout_list)
        stderr = "".join(stderr_list)

        total_dyn_inst: int | None = None

        if profile:
            for line in stderr.splitlines():
                match_ = BRILI_PROFILE_REGEX.match(line)
                if match_ is not None:
                    total_dyn_inst = int(match_.groups()[0])
                    break

        return BriliOutput(
            returncode=returncode,
            stdout="".join(stdout),
            stderr="".join(stderr),
            error=error,
            total_dyn_inst=total_dyn_inst,
        )


BRILI_MAP: dict[str, Brili] = {
    "subprocess": SubprocessBrili(),
    "brilirs": Brilirs(),
}


@click.command()
@click.option(
    "--brili",
    default="subprocess",
    type=click.Choice(BRILI_MAP.keys(), case_sensitive=False),
    help="Which Brili implementation to use",
)
@click.option(
    "-p",
    "--profile",
    is_flag=True,
    default=False,
    type=bool,
    help="Profile total number of dynamic instructions",
)
@click.argument("arguments", nargs=-1)
def main(brili: str, profile: bool, arguments: list[str]):
    brili_impl = BRILI_MAP[brili]
    program: Program = json.load(sys.stdin)
    print(brili_impl.interpret_json(cast(dict, program), arguments, profile=profile))


if __name__ == "__main__":
    main()
