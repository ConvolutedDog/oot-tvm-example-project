import tvm
from tvm import relax as rx
from tvm.script import relax as R

from tvm import tir

test_cases = {}


def register(func):
    test_cases[func.__name__] = func
    return func


@register
def test_var():
    v0 = rx.Var("v0", R.Tensor((1, 2, 3), "float32"))
    assert v0.name_hint == "v0"
    assert v0.struct_info == R.Tensor((1, 2, 3), "float32")


@register
def test_dataflow_var():
    v0 = rx.DataflowVar("v0", R.Tensor((1, 2, 3), "float32"))
    assert v0.name_hint == "v0"
    assert v0.struct_info == R.Tensor((1, 2, 3), "float32")


@register
def test_match_cast():

    # rx.MatchCast(var,value,struct_info)
    # if `value` match `struct_info`, cast `value`'s struct_info into `struct_info` in runtime,
    # then assign `value` to `var`.
    # rx.MatchCast is mainly used to dynamic shape inference
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")

    x = rx.Var("x", R.Tensor([m, n], "float32"))
    y = rx.MatchCast(rx.Var("y"), x, R.Tensor([10, 10], "float32"))

    assert y.struct_info == R.Tensor([10, 10], "float32")

    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("b0"), shape, R.Tensor([m, n], "int32"))
    assert b0.struct_info == R.Tensor([m, n], "int32")

    value = rx.Var("value", R.Tensor(None, "float32", ndim=-1))
    var = rx.Var("var", R.Tensor([m, n], "float32"))
    b1 = rx.MatchCast(var, value, R.Tensor([10, 10], "float32"))

    assert b1.value == value
    assert b1.struct_info == R.Tensor([10, 10], "float32")


@register
def test_var_binding():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")

    import numpy as np
    # binding a value to a var

    # rx.const support numpy array as arguments
    value1 = rx.const(np.random.rand(24, 56))
    bind1 = rx.VarBinding(rx.Var("bind1"), value1)

    assert bind1.var.name_hint == "bind1"
    assert bind1.value == value1

    shape = rx.const(np.array([16, 8]), "int32")
    bind2 = rx.MatchCast(rx.Var("bind2"), shape, R.Tensor([m, n], "int32"))
    assert bind2.struct_info == R.Tensor([m, n], "int32")
    assert bind2.value == shape
    # rx.MatchCast is also a relax.Binding

    block0 = rx.BindingBlock([bind1, bind2])
    assert block0.bindings[0] == bind1
    assert block0.bindings[1] == bind2
    print(block0)


@register
def test_dataflow_block():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")

    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    import numpy as np
    v1 = rx.Var("v1")
    val1 = rx.const([1, 2], "int32")
    b1 = rx.VarBinding(v1, val1)

    block1 = rx.DataflowBlock([b0, b1])
    assert block1.bindings[0] == b0
    assert block1.bindings[1] == b1
    print(block1)


@register
def test_seq_expr():
    x = rx.Var("x", R.Tensor([2, 4], "int32"))
    y = rx.Var("y", R.Tensor([4, 8], "int32"))
    res = rx.Var("ret", R.Tensor(ndim=-1))

    varBind1 = rx.VarBinding(x, rx.Call(tvm.ir.Op.get("relax.add"), [x, x]))
    varBind2 = rx.VarBinding(res, rx.Call(
        tvm.ir.Op.get("relax.multiply"), [x, y]))

    bindBlock = rx.BindingBlock([varBind1, varBind2])

    seq1 = rx.SeqExpr([bindBlock], res)
    assert seq1.body == res
    assert seq1.blocks[0].bindings[0] == varBind1
    assert seq1.blocks[0].bindings[1] == varBind2

    print(seq1)
    print(res.struct_info)


@register
def test_func():
    m, k, n = tvm.tir.Var("m", "int64"), tvm.tir.Var(
        "k", "int64"), tvm.tir.Var("n", "int64")

    a = rx.Var("a", R.Tensor([m, k], "int32"))
    b = rx.Var("b", R.Tensor([k, n], "int32"))

    c = rx.Call(tvm.ir.Op.get("relax.matmul"), [a, b])

    func = rx.Function([a, b], c, R.Tensor(ndim=-1))

    # update the attribute of func
    func = func.with_attr("global_symbol", "func")
    mod = tvm.IRModule.from_expr(func)
    print(mod)
    print(mod["func"])


@register
def test_shape_expr():
    shape = [96, 54]
    v1 = rx.Var("v1", R.Tensor(shape))
    s1 = rx.get_shape_of(v1)
    print(s1)

    shape_expr = rx.ShapeExpr([10, 20])
    print(shape_expr)


@register
def test_prim_value():
    pv0 = rx.PrimValue(1)
    pv1 = rx.PrimValue(tvm.tir.Mul(2, 3))  # R.prim_value(T.Mul(2, 3))
    pv2 = rx.PrimValue(tvm.tir.Var("n", "int32") + 1)  # R.prim_value(n + 1)
    pv3 = rx.PrimValue(tvm.tir.IntImm("int64", 1))


@register
def test_call():
    x = rx.Var("x", R.Tensor(ndim=-1))
    y = rx.Var("y", R.Tensor(ndim=-1))
    z = rx.Call(tvm.ir.Op.get("relax.add"), [x, y])
    m = rx.op.add(x, z)

    func = rx.Function([x, y], m, R.Tensor(ndim=-1))
    print(func)


if __name__ == "__main__":
    for name, func in test_cases.items():
        print(f"⭕⭕⭕Running test case: {name}")
        func()
        print(f"✅✅✅Test case {name} passed")
