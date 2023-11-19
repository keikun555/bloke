from sys import stdin
from typing import NoReturn


async def read_stdin() -> str:
    data = stdin.read()
    return data


def unreachable(x: NoReturn) -> NoReturn:
    raise Exception("impossible case reached")


# Usage example
if __name__ == "__main__":
    input_data = read_stdin()
    print(input_data)
