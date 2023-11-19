"""
Overflow doesn't work as similar to brili
"""
import copy
import json
import math
import sys
from typing import Any, Generic, Literal, NamedTuple, Optional, TypedDict, TypeVar, cast

import numpy as np
from typing_extensions import NotRequired

from brili import bril
from brili.util import unreachable


def bril_int_get(val: int) -> int:
    # https://stackoverflow.com/questions/7770949/simulating-integer-overflow-in-python
    size = 64
    sign = 1 << size - 1

    signed = (val & sign - 1) - (val & sign)
    return signed


class BriliError(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message)
        self.name = self.__class__.__name__


def error(message: str) -> BriliError:
    return BriliError(message)


class Key:
    def __init__(self, b: int, o: int):
        self.base = b
        self.offset = o

    def add(self, offset: int) -> "Key":
        return Key(self.base, self.offset + offset)


X = TypeVar("X")


class Heap(Generic[X]):
    def __init__(self):
        self.storage: dict[int, list[X | None]] = {}

    def is_empty(self) -> bool:
        return len(self.storage) == 0

    count = 0

    def get_new_base(self) -> int:
        val = self.count
        self.count += 1
        return val

    def free_key(self, key: Key) -> None:
        return

    def alloc(self, amt: int) -> Key:
        if amt <= 0:
            raise error(f"cannot allocate {amt} entries")
        base = self.get_new_base()
        self.storage[base] = [None] * amt
        return Key(base, 0)

    def free(self, key: Key) -> None:
        if key.base in self.storage and key.offset == 0:
            self.free_key(key)
            del self.storage[key.base]
        else:
            raise error(
                f"Tried to free illegal memory location base: {key.base}, offset: {key.offset}. Offset must be 0."
            )

    def write(self, key: Key, val: X) -> None:
        data = self.storage.get(key.base)
        if data and len(data) > key.offset and key.offset >= 0:
            data[key.offset] = val
        else:
            raise error(
                f"Uninitialized heap location {key.base} and/or illegal offset {key.offset}"
            )

    def read(self, key: Key) -> X:
        data = self.storage.get(key.base)
        if (
            data
            and len(data) > key.offset
            and key.offset >= 0
            and data[key.offset] is not None
        ):
            return cast(X, data[key.offset])
        raise error(
            f"Uninitialized heap location {key.base} and/or illegal offset {key.offset}"
        )


argCounts: dict[str, int | None] = {
    "add": 2,
    "mul": 2,
    "sub": 2,
    "div": 2,
    "id": 1,
    "lt": 2,
    "le": 2,
    "gt": 2,
    "ge": 2,
    "eq": 2,
    "not": 1,
    "and": 2,
    "or": 2,
    "fadd": 2,
    "fmul": 2,
    "fsub": 2,
    "fdiv": 2,
    "flt": 2,
    "fle": 2,
    "fgt": 2,
    "fge": 2,
    "feq": 2,
    "print": None,  # Any number of arguments.
    "br": 1,
    "jmp": 0,
    "ret": None,  # (Should be 0 or 1.)
    "nop": 0,
    "call": None,
    "alloc": 1,
    "free": 1,
    "store": 2,
    "load": 1,
    "ptradd": 2,
    "phi": None,
    "speculate": 0,
    "guard": 1,
    "commit": 0,
    "ceq": 2,
    "clt": 2,
    "cle": 2,
    "cgt": 2,
    "cge": 2,
    "char2int": 1,
    "int2char": 1,
}


class Pointer:
    def __init__(self, loc: Key, type: bril.Type):
        self.loc = loc
        self.type = type


Value = bool | int | Pointer | float | str
Env = dict[str, Value]


def typeCheck(val: Value, typ: bril.Type) -> bool:
    if typ == "int":
        return isinstance(val, int)
    elif typ == "bool":
        return isinstance(val, bool)
    elif typ == "float":
        return isinstance(val, float)
    elif isinstance(typ, dict) and "ptr" in typ:
        return isinstance(val, Pointer)
    elif typ == "char":
        return isinstance(val, str)
    raise error(f"unknown type {typ}")


def typeCmp(lhs: bril.Type, rhs: bril.Type) -> bool:
    if not isinstance(lhs, dict) and lhs in {"int", "bool", "float", "char"}:
        return lhs == rhs
    else:
        if (
            isinstance(rhs, dict)
            and "ptr" in rhs
            and isinstance(lhs, dict)
            and "ptr" in lhs
        ):
            return typeCmp(lhs["ptr"], rhs["ptr"])
        else:
            return False


def get(env: Env, ident: bril.Ident) -> Value:
    val = env.get(ident)
    if val is None:
        raise error(f"undefined variable {ident}")
    return val


def findFunc(func: bril.Ident, funcs: list[bril.Function]) -> bril.Function:
    matches = [f for f in funcs if f["name"] == func]

    if len(matches) == 0:
        raise error(f"no function of name {func} found")
    elif len(matches) > 1:
        raise error(f"multiple functions of name {func} found")

    return matches[0]


def alloc(ptrType: bril.ParamType, amt: int, heap: Heap[Value]) -> Pointer:
    if not isinstance(ptrType, dict):
        raise error(f"unspecified pointer type {ptrType}")
    elif amt <= 0:
        raise error(f"must allocate a positive amount of memory: {amt} <= 0")
    else:
        loc = heap.alloc(amt)
        dataType = ptrType["ptr"]
        return Pointer(loc, dataType)


def checkArgs(instr: bril.Operation, count: int) -> None:
    found = len(instr["args"]) if "args" in instr else 0
    if found != count:
        raise error(f"{instr['op']} takes {count} argument(s); got {found}")


def getPtr(instr: bril.Operation, env: Env, index: int) -> Pointer:
    val = getArgument(instr, env, index)
    if not isinstance(val, Pointer):
        raise error(f"{instr['op']} argument {index} must be a Pointer")
    return val


def getArgument(
    instr: bril.Operation, env: Env, index: int, typ: bril.Type | None = None
) -> Value:
    args = instr.get("args", [])
    if len(args) <= index:
        raise error(
            f"{instr['op']} expected at least {index+1} arguments; got {len(args)}"
        )
    val = get(env, args[index])
    if typ and not typeCheck(val, typ):
        raise error(f"{instr['op']} argument {index} must be a {typ}")
    return val


def getInt(instr: bril.Operation, env: Env, index: int) -> int:
    return cast(int, getArgument(instr, env, index, "int"))


def getBool(instr: bril.Operation, env: Env, index: int) -> bool:
    return cast(bool, getArgument(instr, env, index, "bool"))


def getFloat(instr: bril.Operation, env: Env, index: int) -> float:
    return cast(float, getArgument(instr, env, index, "float"))


def getChar(instr: bril.Operation, env: Env, index: int) -> str:
    return cast(str, getArgument(instr, env, index, "char"))


def getLabel(instr: bril.Operation, index: int) -> bril.Ident:
    if not instr["labels"]:
        raise error(f"missing labels; expected at least {index+1}")
    if len(instr["labels"]) <= index:
        raise error(f"expecting {index+1} labels; found {len(instr['labels'])}")
    return instr["labels"][index]


def getFunc(instr: bril.Operation, index: int) -> bril.Ident:
    if not instr["funcs"]:
        raise error(f"missing functions; expected at least {index+1}")
    if len(instr["funcs"]) <= index:
        raise error(f"expecting {index+1} functions; found {len(instr['funcs'])}")
    return instr["funcs"][index]


class Action(TypedDict):
    action: Literal["next", "jump", "end", "speculate", "commit", "abort"]
    label: NotRequired[bril.Ident]
    ret: NotRequired[Value | None]


NEXT: Action = {"action": "next"}  # type: ignore


class State:
    def __init__(
        self,
        env: Env,
        heap: Heap[Value],
        funcs: list[bril.Function],
        icount: int,
        curlabel: str | None = None,
        lastlabel: str | None = None,
        specparent: Optional["State"] = None,
    ):
        self.env = env
        self.heap = heap
        self.funcs = funcs
        self.icount = icount
        self.curlabel = curlabel
        self.lastlabel = lastlabel
        self.specparent = specparent


def evalCall(instr: bril.Operation, state: State) -> Action:
    # Which function are we calling?
    funcName = getFunc(instr, 0)
    func = findFunc(funcName, state.funcs)
    if func is None:
        raise error(f"undefined function {funcName}")

    newEnv: Env = {}

    # Check arity of arguments and definition.
    params = func.get("args", [])
    args = instr.get("args", [])
    if len(params) != len(args):
        raise error(f"function expected {len(params)} arguments, got {len(args)}")

    for i in range(len(params)):
        # Look up the variable in the current (calling) environment.
        value = get(state.env, args[i])

        # Check argument types
        if not typeCheck(value, params[i]["type"]):
            raise error("function argument type mismatch")

        # Set the value of the arg in the new (function) environment.
        newEnv[params[i]["name"]] = value

    # Invoke the interpreter on the function.
    newState = State(
        env=newEnv,
        heap=state.heap,
        funcs=state.funcs,
        icount=state.icount,
        lastlabel=None,
        curlabel=None,
        specparent=None,  # Speculation not allowed.
    )
    retVal = evalFunc(func, newState)
    state.icount = newState.icount

    # Dynamically check the function's return value and type.
    if "dest" not in instr:  # `instr` is an `EffectOperation`.
        # Expected void function
        if retVal is not None:
            raise error("unexpected value returned without destination")
        if "type" in func:
            raise error(
                f"non-void function (type: {func['type']}) doesn't return anything"
            )
    else:  # `instr` is a `ValueOperation`.
        instr = cast(bril.ValueOperation, instr)
        # Expected non-void function
        if "type" not in instr:
            raise error("function call must include a type if it has a destination")
        if "dest" not in instr:
            raise error("function call must include a destination if it has a type")
        if retVal is None:
            raise error(
                f"non-void function (type: {func['type']}) doesn't return anything"
            )
        if not typeCheck(retVal, instr["type"]):
            raise error(
                "type of value returned by function does not match destination type"
            )
        if "type" not in func:
            raise error("function with void return type used in value call")
        if not typeCmp(instr["type"], func["type"]):
            raise error("type of value returned by function does not match declaration")
        state.env[instr["dest"]] = retVal

    return NEXT


def evalInstr(instr: bril.Instruction, state: State) -> Action:
    state.icount += 1

    # Check that we have the right number of arguments.
    if instr["op"] != "const":
        if instr["op"] not in argCounts:
            raise error("unknown opcode " + instr["op"])
        count = argCounts[instr["op"]]
        if count is not None:
            checkArgs(instr, count)

    # Function calls are not (currently) supported during speculation.
    # It would be cool to add, but aborting from inside a function call
    # would require explicit stack management.
    if state.specparent and instr["op"] in ["call", "ret"]:
        raise error(f"{instr['op']} not allowed during speculation")

    lhs: int | float
    rhs: int | float

    if instr["op"] == "const":
        # Interpret JSON numbers as either ints or floats.
        value: Value
        if isinstance(instr["value"], bool):
            value = bool(instr["value"])
        elif isinstance(instr["value"], (int, float)):
            if instr["type"] == "float":
                value = float(instr["value"])
            else:
                value = bril_int_get(math.floor(instr["value"]))
        elif isinstance(instr["value"], str):
            if len(instr["value"]) != 1:
                raise error("char must have one character")
            value = instr["value"]
        else:
            value = instr["value"]

        state.env[instr["dest"]] = value
        return NEXT

    elif instr["op"] == "id":
        val = getArgument(instr, state.env, 0)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] in ["add", "mul", "sub"]:
        instr = cast(bril.ValueOperation, instr)
        val = getInt(instr, state.env, 0)
        if instr["op"] == "add":
            val += getInt(instr, state.env, 1)
        elif instr["op"] == "mul":
            val *= getInt(instr, state.env, 1)
        elif instr["op"] == "sub":
            val -= getInt(instr, state.env, 1)
        val = bril_int_get(val)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] in ["fadd", "fmul", "fsub"]:
        instr = cast(bril.ValueOperation, instr)
        val = getFloat(instr, state.env, 0)
        if instr["op"] == "fadd":
            val += getFloat(instr, state.env, 1)
        elif instr["op"] == "fmul":
            val *= getFloat(instr, state.env, 1)
        elif instr["op"] == "fsub":
            val -= getFloat(instr, state.env, 1)
        val = float(val)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "div":
        instr = cast(bril.ValueOperation, instr)
        lhs = getInt(instr, state.env, 0)
        rhs = getInt(instr, state.env, 1)
        if rhs == 0:
            raise error("division by zero")
        val = lhs / rhs
        val = bril_int_get(cast(int, val))
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] in ["le", "lt", "ge", "gt", "eq"]:
        instr = cast(bril.ValueOperation, instr)
        if instr["op"] == "le":
            val = getInt(instr, state.env, 0) <= getInt(instr, state.env, 1)
        elif instr["op"] == "lt":
            val = getInt(instr, state.env, 0) < getInt(instr, state.env, 1)
        elif instr["op"] == "ge":
            val = getInt(instr, state.env, 0) >= getInt(instr, state.env, 1)
        elif instr["op"] == "gt":
            val = getInt(instr, state.env, 0) > getInt(instr, state.env, 1)
        elif instr["op"] == "eq":
            val = getInt(instr, state.env, 0) == getInt(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "not":
        val = not getBool(instr, state.env, 0)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] in ["and", "or"]:
        instr = cast(bril.ValueOperation, instr)
        if instr["op"] == "and":
            val = getBool(instr, state.env, 0) and getBool(instr, state.env, 1)
        elif instr["op"] == "or":
            val = getBool(instr, state.env, 0) or getBool(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "fdiv":
        instr = cast(bril.ValueOperation, instr)
        lhs = getFloat(instr, state.env, 0)
        rhs = getFloat(instr, state.env, 1)
        if rhs == 0:
            val = float("inf")
        else:
            val = lhs / rhs
        val = float(val)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "fle":
        val = getFloat(instr, state.env, 0) <= getFloat(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "flt":
        val = getFloat(instr, state.env, 0) < getFloat(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "fge":
        val = getFloat(instr, state.env, 0) >= getFloat(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "fgt":
        val = getFloat(instr, state.env, 0) > getFloat(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "feq":
        val = getFloat(instr, state.env, 0) == getFloat(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "print":
        args = instr.get("args", [])
        values = [val for val in (get(state.env, i) for i in args)]
        values = [
            f"{val:.17f}" if isinstance(val, float) else str(val).lower()
            for val in values
        ]
        print(*values)
        return NEXT

    elif instr["op"] == "jmp":
        return {"action": "jump", "label": getLabel(instr, 0)}

    elif instr["op"] == "br":
        cond = getBool(instr, state.env, 0)
        return (
            {"action": "jump", "label": getLabel(instr, 0)}
            if cond
            else {"action": "jump", "label": getLabel(instr, 1)}
        )

    elif instr["op"] == "ret":
        args = instr.get("args", [])
        if len(args) == 0:
            return {"action": "end", "ret": None}
        elif len(args) == 1:
            val = get(state.env, args[0])
            return {"action": "end", "ret": val}
        else:
            raise error(f"ret takes 0 or 1 argument(s); got {len(args)}")

    elif instr["op"] == "nop":
        return NEXT

    elif instr["op"] == "call":
        return evalCall(instr, state)

    elif instr["op"] == "alloc":
        amt = getInt(instr, state.env, 0)
        typ = instr["type"]
        if not isinstance(typ, dict) or "ptr" not in typ:
            raise error(f"cannot allocate non-pointer type {instr['type']}")
        ptr = alloc(typ, int(amt), state.heap)
        state.env[instr["dest"]] = ptr
        return NEXT

    elif instr["op"] == "free":
        val = getPtr(instr, state.env, 0)
        state.heap.free(val.loc)
        return NEXT

    elif instr["op"] == "store":
        target = getPtr(instr, state.env, 0)
        state.heap.write(target.loc, getArgument(instr, state.env, 1, target.type))
        return NEXT

    elif instr["op"] == "load":
        ptr = getPtr(instr, state.env, 0)
        val = state.heap.read(ptr.loc)
        if val is None:
            raise error(f"Pointer {instr['args'][0]} points to uninitialized data")
        else:
            state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "ptradd":
        ptr = getPtr(instr, state.env, 0)
        val = getInt(instr, state.env, 1)
        state.env[instr["dest"]] = Pointer(loc=ptr.loc.add(int(val)), type=ptr.type)
        return NEXT

    elif instr["op"] == "phi":
        labels = instr.get("labels", [])
        args = instr.get("args", [])
        if len(labels) != len(args):
            raise error("phi node has unequal numbers of labels and args")
        if not state.lastlabel:
            raise error("phi node executed with no last label")
        idx = labels.index(state.lastlabel)
        if idx == -1:
            # Last label not handled. Leave uninitialized.
            if instr["dest"] in state.env:
                del state.env[instr["dest"]]
        else:
            # Copy the right argument (including an undefined one).
            if idx >= len(args):
                raise error(f"phi node needed at least {idx + 1} arguments")
            src = args[idx]
            phi_val = state.env.get(src)
            if phi_val is None:
                if instr["dest"] in state.env:
                    del state.env[instr["dest"]]
            else:
                state.env[instr["dest"]] = phi_val
        return NEXT

    # Begin speculation.
    elif instr["op"] == "speculate":
        return {"action": "speculate"}

    # Abort speculation if the condition is false.
    elif instr["op"] == "guard":
        if getBool(instr, state.env, 0):
            return NEXT
        else:
            return {"action": "abort", "label": getLabel(instr, 0)}

    # Resolve speculation, making speculative state real.
    elif instr["op"] == "commit":
        return {"action": "commit"}

    elif instr["op"] == "ceq":
        val = getChar(instr, state.env, 0) == getChar(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "clt":
        val = getChar(instr, state.env, 0) < getChar(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "cle":
        val = getChar(instr, state.env, 0) <= getChar(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "cgt":
        val = getChar(instr, state.env, 0) > getChar(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "cge":
        val = getChar(instr, state.env, 0) >= getChar(instr, state.env, 1)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "char2int":
        code = ord(getChar(instr, state.env, 0))
        val = bril_int_get(code)
        state.env[instr["dest"]] = val
        return NEXT

    elif instr["op"] == "int2char":
        i = getInt(instr, state.env, 0)
        if i > 1114111 or i < 0 or (55295 < i < 57344):
            raise error(f"value {i} cannot be converted to char")
        val = chr(i)
        state.env[instr["dest"]] = val
        return NEXT

    unreachable(instr)  # type: ignore
    raise error(f"unhandled opcode {instr['op']}")


def evalFunc(func: bril.Function, state: State) -> Value | None:
    i = 0
    while i < len(func["instrs"]):
        line = func["instrs"][i]
        if "op" in line:
            line = cast(bril.Instruction, line)
            # Run an instruction.
            action = evalInstr(line, state)

            # Take the prescribed action.
            if action["action"] == "end":
                # Return from this function.
                return action["ret"]
            elif action["action"] == "speculate":
                # Begin speculation.
                state.specparent = copy.deepcopy(state)
                state.env = dict(state.env)
            elif action["action"] == "commit":
                # Resolve speculation.
                if not state.specparent:
                    raise error("commit in non-speculative state")
                state.specparent = None
            elif action["action"] == "abort":
                # Restore state.
                if not state.specparent:
                    raise error("abort in non-speculative state")
                # We do *not* restore `icount` from the saved state to ensure that we
                # count "aborted" instructions.
                state.specparent = state.specparent
                state.env = dict(state.specparent.env)
                state.lastlabel = state.specparent.lastlabel
                state.curlabel = state.specparent.curlabel

            # Move to a label.
            if "label" in action:
                # Search for the label and transfer control.
                for j in range(len(func["instrs"])):
                    sLine = func["instrs"][j]
                    if (
                        "label" in sLine
                        and cast(bril.Label, sLine)["label"] == action["label"]
                    ):
                        i = j - 1  # Execute the label next.
                        break
                if i == len(func["instrs"]) - 1:
                    raise error(f"label {action['label']} not found")
        elif "label" in line:
            # Update CFG tracking for SSA phi nodes.
            state.lastlabel = state.curlabel
            state.curlabel = cast(bril.Label, line)["label"]

        i += 1

    # Reached the end of the function without hitting `ret`.
    if state.specparent:
        raise error("implicit return in speculative state")
    return None


def parseChar(s: str) -> str:
    c = s
    if len(c) == 1:
        return c
    else:
        raise error(f"char argument to main must have one character; got {s}")


def parseBool(s: str) -> bool:
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise error(f"boolean argument to main must be 'true'/'false'; got {s}")


def parseNumber(s: str) -> float:
    f = float(s)
    f2 = float(s)
    if not math.isnan(f) and f == f2:
        return f
    else:
        raise error(f"float argument to main must not be 'NaN'; got {s}")


def parseMainArguments(expected: list[bril.Argument], args: list[str]) -> Env:
    newEnv: Env = {}

    if len(args) != len(expected):
        raise error(
            f"mismatched main argument arity: expected {len(expected)}; got {len(args)}"
        )

    for i in range(len(args)):
        type_ = expected[i]["type"]
        if type_ == "int":
            n: int = int(args[i])
            newEnv[expected[i]["name"]] = n
        elif type_ == "float":
            f: float = parseNumber(args[i])
            newEnv[expected[i]["name"]] = f
        elif type_ == "bool":
            b: bool = parseBool(args[i])
            newEnv[expected[i]["name"]] = b
        elif type_ == "char":
            c: str = parseChar(args[i])
            newEnv[expected[i]["name"]] = c

    return newEnv


def evalProg(prog: bril.Program) -> None:
    heap = Heap[Value]()
    main_func = findFunc("main", prog["functions"])
    if main_func is None:
        print("no main function defined, doing nothing", file=sys.stderr)
        return

    # Silly argument parsing to find the `-p` flag.
    args: list[str] = sys.argv[1:]
    profiling = False
    if "-p" in args:
        profiling = True
        args.remove("-p")

    # Remaining arguments are for the main function.
    expected = main_func.get("args", [])
    newEnv = parseMainArguments(expected, args)

    state: State = State(
        funcs=prog["functions"],
        heap=heap,
        env=newEnv,
        icount=0,
        lastlabel=None,
        curlabel=None,
        specparent=None,
    )
    evalFunc(main_func, state)

    if not heap.is_empty():
        raise error(
            "Some memory locations have not been freed by the end of execution."
        )

    if profiling:
        print(f"total_dyn_inst: {state.icount}", file=sys.stderr)


def main():
    sys.setrecursionlimit(10000)
    # bril_int_get(9223372036854775807 * 3)
    try:
        prog: bril.Program = json.loads(
            sys.stdin.read()
        )  # Use json.loads to parse JSON from stdin
        evalProg(prog)
    except BriliError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
