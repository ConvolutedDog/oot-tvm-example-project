import tvm
import tvm.relax as rx
import tvm.script.relax as R

def struct_info_test():
    s0 = rx.ObjectStructInfo()
    print(tvm.ir.save_json(s0)) 

    s0 = rx.ShapeStructInfo([1, 2, 3])
    assert s0.ndim == 3

    t0 = rx.TensorStructInfo([1, 2, 3], "float32")
    assert t0.ndim == 3
    assert t0.dtype == "float32"
    print(t0.shape) #R.shape([1,2,3])
    # NOTE can't compare `ShapeExpr` as follows.
    # there is no `__eq__` method in `ShapeExpr`
    # assert t0.shape == R.shape([1, 2, 3])

    assert list(t0.shape.values) == [1, 2, 3]

    shapeVar = rx.Var("shape",rx.ShapeStructInfo(ndim=3))
    t1 = rx.TensorStructInfo(shapeVar, "float32")
    assert t1.ndim == 3
    assert t1.dtype == "float32"
    assert t1.shape == shapeVar

    t2 = rx.TupleStructInfo([t0, t1])
    assert t2.fields[0] == t0
    assert t2.fields[1] == t1

    m = tvm.tir.Var("m", "int64")
    n = tvm.tir.Var("n", "int64")
    k = tvm.tir.Var("k", "int64")

    a = rx.TensorStructInfo([m,k], "float32")
    b = rx.TensorStructInfo([k,n], "float32")
    c = rx.TensorStructInfo([m,n], "float32")

    f = rx.FuncStructInfo([a,b], c)
    print(f)

    f1 = rx.FuncStructInfo.opaque_func()
    print(f1)


if __name__ == '__main__':
    struct_info_test()