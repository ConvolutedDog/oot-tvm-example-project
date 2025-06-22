import tvm
from tvm.relax import Expr, PrimValue, StringImm, Tuple
from typing import List
from tvm.relax.type_converter import args_converter

# NOTE `_ArgsConverter` is a class that automates the conversion of Python type arguments into
# TVM Relax Expr or List[Expr].

# tvm.PrimExpr -> relax.PrimValue
# tvm.String or str -> relax.StringImm
# tuple/list of PrimExpr -> relax.Tuple


def test_args_to_expr(prim_value: PrimValue, string_imm: StringImm, tuple: Tuple):
    assert isinstance(prim_value, PrimValue)
    assert isinstance(string_imm, StringImm)
    assert isinstance(tuple, Tuple)
    print(
        f"prim_value: {prim_value}, string_imm: {string_imm}, tuple: {tuple}")


@args_converter.to_expr("prim_value", "string_imm", "tuple")
def test_args_to_expr_decorator(prim_value: PrimValue, string_imm: StringImm, tuple: Tuple):
    assert isinstance(prim_value, PrimValue)
    assert isinstance(string_imm, StringImm)
    assert isinstance(tuple, Tuple)
    print(
        f"prim_value: {prim_value}, string_imm: {string_imm}, tuple: {tuple}")


@args_converter.to_list_expr("prim_value", "string_imm", "tuple")
def test_args_to_list_expr(prim_value: List[PrimValue], string_imm: List[StringImm], tuple: List[Tuple]):
    assert isinstance(prim_value, List) and all(
        [isinstance(arg, PrimValue) for arg in prim_value])
    assert isinstance(string_imm, List) and all(
        [isinstance(arg, StringImm) for arg in string_imm])
    assert isinstance(tuple, List) and all(
        [isinstance(arg, Tuple) for arg in tuple])
    print(
        f"prim_value: {prim_value}, string_imm: {string_imm}, tuple: {tuple}")


# NOTE We can use `args_converter.auto` to automatically convert the arguments without specifying the argument names.
# But we must specify the types of the formal arguments to Expr or List[Expr].
@args_converter.auto
def test_auto_to_expr(prim_value: Expr, string_imm: Expr, tuple: Expr):
    assert isinstance(prim_value, PrimValue)
    assert isinstance(string_imm, StringImm)
    assert isinstance(tuple, Tuple)
    print(f"prim_value: {prim_value}, string_imm: {string_imm}, tuple: {tuple}")

@args_converter.auto
def test_auto_to_list_expr(prim_value: List[Expr], string_imm: List[Expr], tuple: List[Expr]):
    assert isinstance(prim_value, List) and all(
        [isinstance(arg, PrimValue) for arg in prim_value])
    assert isinstance(string_imm, List) and all(
        [isinstance(arg, StringImm) for arg in string_imm])
    assert isinstance(tuple, List) and all(
        [isinstance(arg, Tuple) for arg in tuple])
    
    print(prim_value)
    print(string_imm)
    print(tuple)


if __name__ == "__main__":
    # Some variables with python types
    prim_value = 1
    string_imm = "hello"
    tuple = (1, 2, 3)

    # This will raise an AssertionError because the arguments can't be converted to relax.PrimValue, relax.StringImm, and relax.Tuple.
    # test_args_to_expr(prim_value, string_imm, tuple)

    test_args_to_expr2 = args_converter.to_expr(
        "prim_value", "string_imm", "tuple")(test_args_to_expr)
    # Now `test_args_to_expr2` can implicitly convert its arguments with the python types to relax.PrimValue, relax.StringImm, and relax.Tuple.
    test_args_to_expr2(prim_value, string_imm, tuple)

    # We also can use the decorator to achieve the same effect.
    test_args_to_expr_decorator(prim_value, string_imm, tuple)

    # Test list of arguments
    prim_value = [1, 2, 3]
    string_imm = ["hello", "world"]
    tuple = [(1, 2, 3), (4, 5, 6)]
    test_args_to_list_expr(prim_value, string_imm, tuple)

    print("⭕test_auto_to_list_expr")
    test_auto_to_list_expr(prim_value, string_imm, tuple)
