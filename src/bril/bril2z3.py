"""Converts Loop-free Bril to Z3"""

import copy
import json
import logging
import operator
import sys
from types import MappingProxyType
from typing import Callable, Generator, NamedTuple, TypeAlias, TypedDict, TypeVar, cast

import click
import z3

from bril.basic_blocks import (
    BasicBlock,
    BasicBlockFunction,
    BasicBlockProgram,
    basic_block_program_from_program,
)
from bril.bril_extract import phi_nodes_get, var_to_type_dict_get
from bril.bril_labeler import index_to_label_dict_get, label_to_index_dict_get
from bril.cfg import control_flow_graph_from_instructions, is_cyclic
from bril.typing_bril import (
    Argument,
    BrilType,
    Constant,
    Effect,
    Instruction,
    Operation,
    Program,
    Value,
    Variable,
)

logger = logging.getLogger(__name__)

Z3_RETURN_PREFIX = "BRIL.RETURN"
Z3_PRINT_PREFIX = "BRIL.PRINT.LINES"


def z3_return_arg_name_get(label: int | str) -> str:
    """Get name of return Z3 argument"""
    return f"{Z3_RETURN_PREFIX}.{label}"


ArgType = TypeVar("ArgType")
RetType = TypeVar("RetType")
UnaryOperator: TypeAlias = Callable[[ArgType], RetType]
BinaryOperator: TypeAlias = Callable[[ArgType, ArgType], RetType]


class BrilAnyType(z3.DatatypeSortRef):
    """Models any type arguments such as return and print arguments"""

    int_ref: "BrilAnyType"
    float_ref: "BrilAnyType"
    bool_ref: "BrilAnyType"
    nil: "BrilAnyType"

    def IntV(self, int_ref):
        ...

    def FloatV(self, float_ref):
        ...

    def BoolV(self, bool_ref):
        ...


def z3_bril_any_type_get() -> BrilAnyType:
    """Construct any argument type sorts"""
    any_type = z3.Datatype("AnyType")
    any_type.declare("IntV", ("int", z3.BitVecSort(64)))
    any_type.declare("FloatV", ("float", z3.Float64()))
    any_type.declare("BoolV", ("bool", z3.BoolSort()))
    any_type.declare("nil")
    return cast(BrilAnyType, any_type.create())


BRIL_TYPE_SORT = z3_bril_any_type_get()


def z3_bril_any_var_get(variable: Variable) -> z3.DatatypeSortRef:
    """Z3 any var"""
    return z3.Const(variable, BRIL_TYPE_SORT)


class PhiMaps(NamedTuple):
    """
    phi_dest_to_var: mapping from a phi destination to which variable
                     it should be equal to; could be undefined
    var_to_phi_dest: mapping from a variable to a phi destination
    """

    phi_dest_to_var: dict[Variable, Variable]
    var_to_phi_dest: MappingProxyType[Variable, Variable]

    def copy(self) -> "PhiMaps":
        """Return copy of self"""
        return PhiMaps(copy.deepcopy(self.phi_dest_to_var), self.var_to_phi_dest)


class ProgramState(NamedTuple):
    """Immutable state keeping track of program state"""

    program_label: int
    print_index: int


class FunctionState(NamedTuple):
    """Immutable state keeping track of function state"""

    program_label: int
    print_index: int
    returned: bool  # ret was called
    phi_maps: PhiMaps


class BlockState(TypedDict):
    """Mutable state for executing instructions within a block"""

    program_label: int
    print_index: int
    terminated: bool  # block terminated with a terminator instruction
    returned: bool  # ret was called
    errored: bool  # code errored
    cond: tuple[z3.BoolRef, str, str] | None
    phi_maps: PhiMaps


COMPATIBLE_OPS: tuple[Operation, ...] = (
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
    # "call",
    "ret",
    "id",
    # "print",
    "nop",
    # memory
    # "free",
    # "alloc",
    # "store",
    # "load",
    # "ptradd",
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
    # "speculate",
    # "commit",
    # "guard",
    # character
    # "ceq",
    # "clt",
    # "cle",
    # "cgt",
    # "cge",
    # "char2int",
    # "int2char",
)


def z3_print_variable_name_generator() -> Generator[str, None, None]:
    """Generate print variable names"""
    i = 0
    while True:
        yield f"{Z3_PRINT_PREFIX}.{i}"
        i += 1


def z3_bril_int_get(variable: Variable) -> z3.BitVecRef:
    """Get Z3 bitvector for bril ints"""
    return z3.BitVec(variable, 64)


def z3_bril_bool_get(variable: Variable) -> z3.BoolRef:
    """Get Z3 Bool for bril bools"""
    return z3.Bool(variable)


def z3_bril_float_get(variable: Variable) -> z3.FPRef:
    """Get Z3 FP for bril floats"""
    return z3.FP(variable, z3.Float64())


def z3_bril_variable_get(variable: Variable, type_: BrilType) -> z3.ExprRef:
    """Get Z3 refs for variables"""
    match type_:
        case "int":
            return z3_bril_int_get(variable)
        case "float":
            return z3_bril_float_get(variable)
        case "bool":
            return z3_bril_bool_get(variable)

    raise Exception(f"z3_bril_variable_get, type {type_} not implemented")


def z3_bril_argument_get(argument: Argument) -> z3.ExprRef:
    """Get Z3 refs for arguments"""
    return z3_bril_variable_get(argument["name"], argument["type"])


def bril2z3_compatible(func: BasicBlockFunction) -> bool:
    """Return True iff we can run bril2z3 on the function"""
    for block in func["instrs"]:
        for instruction in block:
            if "op" not in instruction:
                continue

            effect = cast(Effect, instruction)
            if effect["op"] not in COMPATIBLE_OPS:
                logger.error("Incompatible operation %s", effect["op"])
                return False

    cfg = control_flow_graph_from_instructions(func["instrs"])

    if is_cyclic(cfg):
        # Loop-free
        logger.error("Loop found")
        return False

    return True


def bril_const_to_z3(constant: Constant) -> z3.ExprRef:
    """Convert Bril const instruction to Z3"""
    return z3_bril_variable_get(constant["dest"], constant["type"]) == constant["value"]


def bril_id_to_z3(value: Value) -> z3.ExprRef:
    """Convert Bril id instruction to Z3"""
    arg1 = z3_bril_variable_get(value["args"][0], value["type"])
    dest_var = z3_bril_variable_get(value["dest"], value["type"])
    return dest_var == arg1


def bril_phi_to_z3(value: Value, phi_maps: PhiMaps) -> z3.ExprRef | None:
    """Convert Bril phi instruction to Z3"""
    phi_dest = value["dest"]
    if phi_dest not in phi_maps.phi_dest_to_var:
        return None

    var = phi_maps.phi_dest_to_var[phi_dest]

    phi_dest_var = z3_bril_variable_get(phi_dest, value["type"])
    argument = z3_bril_variable_get(var, value["type"])
    return phi_dest_var == argument


def unary_op_to_z3(
    value: Value,
    oper: UnaryOperator[z3.ExprRef, z3.ExprRef],
    arg_constructor: Callable[[Variable], z3.ExprRef],
    value_constructor: Callable[[Variable], z3.ExprRef],
) -> z3.ExprRef:
    """Convert Bril unary operator to Z3"""
    arg1 = value["args"][0]
    z3_expr = value_constructor(value["dest"]) == oper(arg_constructor(arg1))
    return z3_expr


def binary_op_to_z3(
    value: Value,
    oper: BinaryOperator[z3.ExprRef, z3.ExprRef],
    arg_constructor: Callable[[Variable], z3.ExprRef],
    value_constructor: Callable[[Variable], z3.ExprRef],
) -> z3.ExprRef:
    """Convert Bril binary operator to Z3"""
    arg1, arg2 = value["args"]
    z3_expr = value_constructor(value["dest"]) == oper(
        arg_constructor(arg1), arg_constructor(arg2)
    )
    return z3_expr


def value_to_z3(value: Value, state: BlockState) -> z3.ExprRef | None:
    """Generates Z3 formula from a bril value"""
    z3_expr: z3.ExprRef | None
    match value["op"]:
        case "const":
            constant = cast(Constant, value)
            z3_expr = bril_const_to_z3(constant)
        # core
        case "add":
            z3_expr = binary_op_to_z3(
                value, operator.add, z3_bril_int_get, z3_bril_int_get
            )
        case "mul":
            z3_expr = binary_op_to_z3(
                value, operator.mul, z3_bril_int_get, z3_bril_int_get
            )
        case "sub":
            z3_expr = binary_op_to_z3(
                value, operator.sub, z3_bril_int_get, z3_bril_int_get
            )
        case "div":
            # In Bril it's an error to divide by zero
            _, arg2 = value["args"]
            if arg2 == 0:
                state["errored"] = True
            # In Z3 Int / Int is the operator.floordiv
            z3_expr = binary_op_to_z3(
                value, operator.truediv, z3_bril_int_get, z3_bril_int_get
            )
        case "eq":
            z3_expr = binary_op_to_z3(
                value, operator.eq, z3_bril_int_get, z3_bril_bool_get
            )
        case "lt":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.lt),
                z3_bril_int_get,
                z3_bril_bool_get,
            )
        case "gt":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.gt),
                z3_bril_int_get,
                z3_bril_bool_get,
            )
        case "le":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.le),
                z3_bril_int_get,
                z3_bril_bool_get,
            )
        case "ge":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.ge),
                z3_bril_int_get,
                z3_bril_bool_get,
            )
        case "not":
            z3_expr = unary_op_to_z3(value, z3.Not, z3_bril_bool_get, z3_bril_bool_get)
        case "and":
            z3_expr = binary_op_to_z3(value, z3.And, z3_bril_bool_get, z3_bril_bool_get)
        case "or":
            z3_expr = binary_op_to_z3(value, z3.Or, z3_bril_bool_get, z3_bril_bool_get)
        # case "call":
        case "id":
            z3_expr = bril_id_to_z3(value)
        # memory
        # case "alloc":
        # case "load":
        # case "ptradd":
        # float
        case "fadd":
            z3_expr = binary_op_to_z3(
                value, operator.add, z3_bril_float_get, z3_bril_float_get
            )
        case "fmul":
            z3_expr = binary_op_to_z3(
                value, operator.mul, z3_bril_float_get, z3_bril_float_get
            )
        case "fsub":
            z3_expr = binary_op_to_z3(
                value, operator.sub, z3_bril_float_get, z3_bril_float_get
            )
        case "fdiv":
            # In Bril it's an error to divide by zero
            _, arg2 = value["args"]
            if arg2 == 0:
                state["errored"] = True
            z3_expr = binary_op_to_z3(
                value, operator.truediv, z3_bril_int_get, z3_bril_int_get
            )
        case "feq":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.eq),
                z3_bril_float_get,
                z3_bril_bool_get,
            )
        case "flt":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.lt),
                z3_bril_float_get,
                z3_bril_bool_get,
            )
        case "fgt":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.gt),
                z3_bril_float_get,
                z3_bril_bool_get,
            )
        case "fle":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.le),
                z3_bril_float_get,
                z3_bril_bool_get,
            )
        case "fge":
            z3_expr = binary_op_to_z3(
                value,
                cast(BinaryOperator, operator.ge),
                z3_bril_float_get,
                z3_bril_bool_get,
            )
        # SSA
        case "phi":
            phi_dest = value["dest"]
            if phi_dest not in state["phi_maps"].phi_dest_to_var:
                state["errored"] = True
            z3_expr = bril_phi_to_z3(value, state["phi_maps"])
        # speculative execution
        # character
        # case "ceq":
        # case "clt":
        # case "cle":
        # case "cgt":
        # case "cge":
        # case "char2int":
        # case "int2char":
        case _:
            raise NotImplementedError(f"value_to_z3, not implemented: {value}")

    # phi_maps bookkeeping for Phi instructions
    dest = value["dest"]
    if dest in state["phi_maps"].var_to_phi_dest:
        state["phi_maps"].phi_dest_to_var[
            state["phi_maps"].var_to_phi_dest[dest]
        ] = dest

    return z3_expr


def bril_print_to_z3(
    effect: Effect,
    var_to_type: dict[Variable, BrilType],
    print_index: int,
) -> z3.ExprRef:
    """Converts Bril print instruction to Z3"""
    args = effect["args"]
    for arg in args:
        match var_to_type[arg]:
            case "int":
                ...
            case "float":
                ...
            case "bool":
                ...
            # case "char":
            case _:
                raise NotImplementedError(
                    f"bril_print_to_z3, not implemented: {effect}"
                )
    return z3.BoolVal(True)  # TODO


def bril_ret_var_to_z3(label: int | str) -> z3.DatatypeSortRef:
    """Make return variable Z3 reference"""
    return_var_name = z3_return_arg_name_get(label)
    return z3_bril_any_var_get(Variable(return_var_name))


def bril_ret_to_z3_eq(return_value: BrilAnyType, label: int | str) -> z3.ExprRef:
    """Convert Z3 return expression in z3"""
    return_var = bril_ret_var_to_z3(label)
    return return_var == return_value


def bril_ret_to_z3(
    effect: Effect, var_to_type: dict[Variable, BrilType], label: int | str
) -> z3.ExprRef | None:
    """Converts Bril ret instruction to Z3"""
    has_args = "args" in effect and len(effect["args"]) >= 1

    if not has_args:
        return bril_ret_to_z3_eq(BRIL_TYPE_SORT.nil, label)

    args = effect["args"]
    return_argument = args[0]
    z3_expr: z3.ExprRef | None
    match var_to_type[return_argument]:
        case "int":
            z3_expr = bril_ret_to_z3_eq(
                BRIL_TYPE_SORT.IntV(z3_bril_int_get(return_argument)), label
            )
        case "float":
            z3_expr = bril_ret_to_z3_eq(
                BRIL_TYPE_SORT.FloatV(z3_bril_float_get(return_argument)), label
            )
        case "bool":
            z3_expr = bril_ret_to_z3_eq(
                BRIL_TYPE_SORT.BoolV(z3_bril_bool_get(return_argument)), label
            )
        # case "char":
        case _:
            raise NotImplementedError(f"bril_ret_to_z3, not implemented: {effect}")
    return z3_expr


def effect_to_z3(
    effect: Effect, var_to_type: dict[Variable, BrilType], state: BlockState
) -> z3.ExprRef | None:
    """Generates Z3 formula from a bril effect"""
    z3_expr: z3.ExprRef | None = None
    match effect["op"]:
        # core
        case "jmp":
            state["terminated"] = True
        case "br":
            if_ = z3_bril_bool_get(effect["args"][0])
            then, else_, *_ = effect["labels"]
            state["cond"] = (if_, then, else_)
            state["terminated"] = True
        # case "call":
        case "ret":
            z3_expr = bril_ret_to_z3(effect, var_to_type, state["program_label"])
            state["terminated"] = True
            state["returned"] = True
        case "print":
            z3_expr = bril_print_to_z3(effect, var_to_type, state["print_index"])
            state["print_index"] += 1
        case "nop":
            z3_expr = None
        # memory
        # case "free":
        # case "store":
        # float
        # SSA
        # speculative execution
        # case "speculate":
        # case "commit":
        # case "guard":
        # character
        case _:
            raise NotImplementedError(f"effect_to_z3, not implemented: {effect}")

    return z3_expr


def instruction_to_z3(
    instruction: Instruction, var_to_type: dict[Variable, BrilType], state: BlockState
) -> z3.ExprRef | None:
    """Generates Z3 formula from a bril instruction"""
    if "op" not in instruction:
        return None

    if "dest" not in instruction:
        effect = cast(Effect, instruction)
        return effect_to_z3(effect, var_to_type, state)

    value = cast(Value, instruction)
    return value_to_z3(value, state)


def block_to_z3(
    block: BasicBlock, block_state: BlockState, var_to_type: dict[Variable, BrilType]
) -> z3.ExprRef:
    """Generate Z3 formula from a bril block"""
    z3_expressions: list[z3.ExprRef] = []
    for instruction in block:
        if (
            z3_expr := instruction_to_z3(instruction, var_to_type, block_state)
        ) is not None:
            z3_expressions.append(z3_expr)

        if block_state["terminated"]:
            return z3.And(*z3_expressions)

        if block_state["errored"]:
            return z3.BoolVal(False)

    return z3.And(*z3_expressions)


def phi_maps_get(func: BasicBlockFunction) -> PhiMaps:
    """Construct PhiMaps given a function"""
    var_to_phi_dest: dict[Variable, Variable] = {}
    for block in func["instrs"]:
        phi_nodes = phi_nodes_get(block)
        for phi_node in phi_nodes:
            for arg in phi_node["args"]:
                var_to_phi_dest[arg] = phi_node["dest"]

    return PhiMaps(
        phi_dest_to_var={},
        var_to_phi_dest=MappingProxyType(var_to_phi_dest),
    )


def function_to_z3(func: BasicBlockFunction, program_state: ProgramState) -> z3.ExprRef:
    """Generates Z3 formula from a bril function"""
    # Need CFG for branch analysis (print, ret, and phi)
    cfg = control_flow_graph_from_instructions(func["instrs"])
    label_to_index_dict: dict[str, int] = label_to_index_dict_get(
        index_to_label_dict_get(func, cfg)
    )
    var_to_type: dict[Variable, BrilType] = var_to_type_dict_get(func)
    phi_maps = phi_maps_get(func)

    def helper(block_index: int, function_state: FunctionState) -> z3.ExprRef:
        if not 0 <= block_index < len(func["instrs"]):
            # If not returned, add nil return
            if not function_state.returned:
                return bril_ret_to_z3_eq(
                    BRIL_TYPE_SORT.nil, function_state.program_label
                )
            return z3.BoolVal(True)

        block = func["instrs"][block_index]
        block_state = BlockState(
            program_label=function_state.program_label,
            print_index=function_state.print_index,
            terminated=False,
            returned=function_state.returned,
            errored=False,
            cond=None,
            phi_maps=function_state.phi_maps.copy(),
        )

        block_expr = block_to_z3(block, block_state, var_to_type)

        if block_state["returned"] or block_state["errored"]:
            return block_expr

        # Successors
        successors = cfg.successors(block_index)
        next_state = FunctionState(
            program_label=block_state["program_label"],
            print_index=block_state["print_index"],
            returned=block_state["returned"],
            phi_maps=block_state["phi_maps"],
        )

        if len(successors) <= 0:
            # Returned without ret instruction, add nil return
            return z3.And(
                block_expr,
                bril_ret_to_z3_eq(BRIL_TYPE_SORT.nil, block_state["program_label"]),
            )

        if block_state["cond"] is None:
            # One successor
            assert len(successors) == 1
            successor = successors[0]
            return z3.And(block_expr, helper(successor, next_state))

        # Conditional branching

        if_, then_label, else_label = block_state["cond"]
        then_index = label_to_index_dict[then_label]
        else_index = label_to_index_dict[else_label]

        return z3.And(
            block_expr,
            z3.If(if_, helper(then_index, next_state), helper(else_index, next_state)),
        )

    initial_state = FunctionState(
        program_label=program_state.program_label,
        print_index=program_state.print_index,
        returned=False,
        phi_maps=phi_maps,
    )

    # So that Z3 will type-check function arguments
    arg_exprs: list[z3.ExprRef] = []
    for arg in func["args"]:
        bril_arg = z3_bril_argument_get(arg)
        arg_exprs.append(bril_arg == bril_arg)  # pylint: disable=comparison-with-itself

    return z3.And(*arg_exprs, helper(0, initial_state))


def program_to_z3(
    program: BasicBlockProgram, program_label: int = 0
) -> z3.ExprRef | None:
    """Convert program to Z3"""
    initial_state = ProgramState(program_label=program_label, print_index=0)
    for func in program["functions"]:
        if func["name"] != "main":
            continue
        if bril2z3_compatible(func):
            return function_to_z3(func, initial_state)
        return None
    return None


@click.command()
@click.option(
    "-s",
    "--simplify",
    is_flag=True,
    default=False,
    type=bool,
    help="Simplify expression",
)
@click.option(
    "-l",
    "--program-label",
    default=0,
    type=int,
    help="Program label",
)
def main(simplify: bool, program_label: int) -> None:
    program: Program = json.load(sys.stdin)
    bb_program: BasicBlockProgram = basic_block_program_from_program(program)

    expression = program_to_z3(bb_program, program_label=program_label)
    if simplify and expression is not None:
        expression = z3.simplify(expression)

    if expression is not None:
        print(expression)
    else:
        print("incompatible")


if __name__ == "__main__":
    main()
