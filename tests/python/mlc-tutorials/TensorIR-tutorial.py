import tvm
import numpy as np
import tvm.relax as relax
import tvm.script
from tvm.script import tir as T

dtype = "float32"
MATRIX_SIZE = 1024


@tvm.script.ir_module
class MyTensorIRModule:
    @T.prim_func
    def mm_relu(
        A: T.Buffer((MATRIX_SIZE, MATRIX_SIZE), dtype),
        B: T.Buffer((MATRIX_SIZE, MATRIX_SIZE), dtype),
        C: T.Buffer((MATRIX_SIZE, MATRIX_SIZE), dtype),
    ):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((MATRIX_SIZE, MATRIX_SIZE), dtype)
        for i, j, k in T.grid(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k], dtype="int32")
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(MATRIX_SIZE, MATRIX_SIZE):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j], dtype="int32")
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))  # relu

    @T.prim_func
    def relu(
        A: T.Buffer((MATRIX_SIZE, MATRIX_SIZE), dtype),
        C: T.Buffer((MATRIX_SIZE, MATRIX_SIZE), dtype),
    ):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        for i, j in T.grid(MATRIX_SIZE, MATRIX_SIZE):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j], dtype="int32")
                C[vi, vj] = T.max(A[vi, vj], T.float32(0))


mod = MyTensorIRModule
mod, params = relax.frontend.detach_params(mod)
mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)

sch = tvm.tir.Schedule(mod)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=[None, 4])  # 32 * 4
sch.mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)

block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)
sch.mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)

sch.decompose_reduction(block_Y, k)
sch.mod.show(
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


# Compute mod
rt_lib = tvm.build(mod, target="llvm")
# print(rt_lib.get_source())
rt_lib["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_np, c_nd.numpy(), rtol=1e-5)


# Compute sch.mod
rt_lib_after_sch = tvm.build(sch.mod, target="llvm")
rt_lib_after_sch["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_np, c_nd.numpy(), rtol=1e-5)


# Compute transform
mod_transformed = transform(mod, jfactor=8)
rt_lib_transformed = tvm.build(mod_transformed, "llvm")
rt_lib_transformed["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_np, c_nd.numpy(), rtol=1e-5)


# Time cost
f_timer_before = rt_lib.time_evaluator(
    func_name="mm_relu", dev=tvm.cpu(), number=3, repeat=1, min_repeat_ms=1000
)
print("Time cost of MyTensorIRModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after_sch.time_evaluator(
    func_name="mm_relu", dev=tvm.cpu(), number=3, repeat=1, min_repeat_ms=1000
)
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)
f_timer_transformed = rt_lib_transformed.time_evaluator(
    func_name="mm_relu", dev=tvm.cpu(), number=3, repeat=1, min_repeat_ms=1000
)
print(
    "Time cost of transformed mod_transformed %g sec"
    % f_timer_transformed(a_nd, b_nd, c_nd).mean
)
f_timer_gpu = rt_lib.time_evaluator("mm_relu", tvm.cuda(1))
print(
    "Time cost of transformed mod_transformed %g sec"
    % f_timer_gpu(a_nd, b_nd, c_nd).mean
)
