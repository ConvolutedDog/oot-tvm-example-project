import tvm
import tvm.te as te
import tvm.relax as relax
import numpy as np


dtype = "float32"
MATRIX_SIZE = 1024


A = te.placeholder(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype, name="A")
B = te.placeholder(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype, name="B")
k = te.reduce_axis(dom=(0, MATRIX_SIZE), name="k")
Y = te.compute(
    shape=(MATRIX_SIZE, MATRIX_SIZE),
    fcompute=lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="Y",
)
C = te.compute(
    shape=(MATRIX_SIZE, MATRIX_SIZE), fcompute=lambda i, j: te.max(Y[i, j], 0), name="C"
)


te_func = te.create_prim_func(ops=[A, B, C]).with_attrs(
    {"global_symbol": "mm_relu", "tir.noalias": True}
)
mod = tvm.IRModule({"mm_relu": te_func})


mod, params = relax.frontend.detach_params(mod)
mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)


def transform(mod, jfactor):
    sch = tvm.tir.Schedule(mod)
    block_Y = sch.get_block("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_block("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod


# Compute numpy gold-model
a_np = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(dtype)
b_np = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(dtype)
c_np = np.maximum(a_np @ b_np, 0)


a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((MATRIX_SIZE, MATRIX_SIZE), dtype=dtype)


# Compute transform
mod_transformed = transform(mod, jfactor=8)
mod_transformed.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)
rt_lib_transformed = tvm.build(mod_transformed, "llvm")
rt_lib_transformed["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_np, c_nd.numpy(), rtol=1e-5)
