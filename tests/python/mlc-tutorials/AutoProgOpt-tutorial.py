import tvm
import tvm.relax as relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T

import numpy as np

import tvm.script


# Recap: Transform a Primitive Tensor Function.
MATRIX_SIZE = 128
dtype = "float32"
RANDOM_SEARCH = False
ITERATE_J_FACTOR = False
Stochastic_Meta_Schedule_MM = True
Default_Meta_Schedule_MM = False
Default_Meta_Schedule_Network = False

a_np = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(dtype)
b_np = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(dtype)
c_np = np.maximum(a_np @ b_np, 0)


a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((MATRIX_SIZE, MATRIX_SIZE), dtype=dtype)


def showmod(mod: IRModule, show_meta=True):
    mod.show(
        black_format=True,
        show_meta=show_meta,
        verbose_expr=True,
        show_object_address=False,
        show_all_struct_info=True,
    )


def buildandtest(mod: IRModule):
    lib = tvm.build(mod, target="llvm")
    # nd_res = lib["main"](a_nd, b_nd, c_nd)
    f_timer_before = lib.time_evaluator("main", tvm.cpu())
    print(
        "Time cost of MyModule: %.3f ms"
        % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000)
    )


@tvm.script.ir_module
class MM:
    @T.prim_func
    def main(
        A: T.Buffer(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype),
        B: T.Buffer(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype),
        C: T.Buffer(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE):
            with T.block("C"):
                vi, vj, vk = T.axis.remap(
                    kinds="SSR", bindings=[i, j, k], dtype="int64"
                )
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


mod = MM
showmod(mod)
buildandtest(mod)


def schedule_mm(sch: tvm.tir.Schedule, jfactor=16):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch


sch = tvm.tir.Schedule(mod)
sch = schedule_mm(sch)
print(sch.trace)
showmod(sch.mod)
buildandtest(sch.mod)


def testjfactor():
    for jfactor in range(1, MATRIX_SIZE):
        print(f"jfactor: {jfactor}, ", end="")

        sch = tvm.tir.Schedule(mod)
        sch = schedule_mm(sch, jfactor=jfactor)

        buildandtest(sch.mod)


if ITERATE_J_FACTOR:
    testjfactor()


# Stochastic Schedule Transformation
def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch


sch = tvm.tir.Schedule(mod)
sch = stochastic_schedule_mm(sch)
showmod(sch.mod)
buildandtest(sch.mod)


# Search Over Stochastic Transformations
def random_search(mod: tvm.IRModule, num_trials=5):
    best_result = None
    best_sch = None

    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target="llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean

        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        print(sch.trace)

        # book keep the best result so far
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch

    return best_sch

if RANDOM_SEARCH:
    sch = random_search(mod)
    print(sch.trace)

"""
meta_schedule is the namespace that comes to support search over a space of
possible transformations. There are many additional things that meta-schedule
do behind the scene:
 - Parallel benchmarking across many processes.
 - Use cost models to avoid benchmarking each time.
 - Evolutionary search on the traces instead of randomly sampling at each time.
"""

from tvm import meta_schedule as ms

if Stochastic_Meta_Schedule_MM:
    database = ms.tune_tir(
        mod=mod,
        target="llvm --num-cores=1",
        max_trials_global=64,
        num_trials_per_iter=64,
        space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
        work_dir="./tune_tmp",
    )

    sch = ms.tir_integration.compile_tir(database, mod, "llvm --num-cores=1")
    sch.trace.show()
    showmod(sch.mod)
    buildandtest(sch.mod)

exit()

# Leverage Default AutoScheduling
if Default_Meta_Schedule_MM:
    database = ms.tune_tir(
        mod=mod,
        target="llvm --num-cores=1",
        max_trials_global=64,
        num_trials_per_iter=64,
        # space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
        work_dir="./tune_tmp",
    )

    sch = ms.tir_integration.compile_tir(database, mod, "llvm --num-cores=1")
    sch.trace.show()
    showmod(sch.mod)
    buildandtest(sch.mod)

# Putting Things Back to End to End Model Execution
if Default_Meta_Schedule_Network:
    import torch
    import torchvision

    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()

    import pickle as pkl

    mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))

    data_nd = tvm.nd.array(img.reshape(1, 784))
    nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

    @tvm.script.ir_module
    class MyModuleMixture:
        @T.prim_func
        def linear0(X: T.Buffer((1, 784), "float32"),
                    W: T.Buffer((128, 784), "float32"),
                    B: T.Buffer((128,), "float32"),
                    Z: T.Buffer((1, 128), "float32")):
            T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
            Y = T.alloc_buffer((1, 128), "float32")
            for i, j, k in T.grid(1, 128, 784):
                with T.block("Y"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]

            for i, j in T.grid(1, 128):
                with T.block("Z"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Z[vi, vj] =  Y[vi, vj] + B[vj]

        @R.function
        def main(x: R.Tensor((1, 784), "float32"),
                w0: R.Tensor((128, 784), "float32"),
                b0: R.Tensor((128,), "float32"),
                w1: R.Tensor((10, 128), "float32"),
                b1: R.Tensor((10,), "float32")):
            with R.dataflow():
                lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, 128), dtype="float32"))
                lv1 = R.call_dps_packed("env.relu", (lv0,), R.Tensor((1, 128), dtype="float32"))
                out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, 10), dtype="float32"))
                R.output(out)
            return out

    @tvm.register_func("env.linear", override=True)
    def torch_linear(x: tvm.nd.NDArray,
                    w: tvm.nd.NDArray,
                    b: tvm.nd.NDArray,
                    out: tvm.nd.NDArray):
        x_torch = torch.from_dlpack(x)
        w_torch = torch.from_dlpack(w)
        b_torch = torch.from_dlpack(b)
        out_torch = torch.from_dlpack(out)
        torch.mm(x_torch, w_torch.T, out=out_torch)
        torch.add(out_torch, b_torch, out=out_torch)

    @tvm.register_func("env.relu", override=True)
    def lnumpy_relu(x: tvm.nd.NDArray,
                    out: tvm.nd.NDArray):
        x_torch = torch.from_dlpack(x)
        out_torch = torch.from_dlpack(out)
        torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)

    MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)

    ex = relax.build(MyModuleWithParams, target="llvm")
    showmod(MyModuleWithParams, show_meta=False)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])

    ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)
    print("MyModuleWithParams time-cost: %g ms" % (ftimer(data_nd).mean * 1000))

    mod_linear = tvm.IRModule.from_expr(MyModuleMixture["linear0"].with_attr("global_symbol", "main"))
    showmod(mod_linear, show_meta=False)

    database = ms.tune_tir(
        mod=mod_linear,
        target="llvm --num-cores=1",
        max_trials_global=64,
        num_trials_per_iter=64,
        work_dir="./tune_tmp",
    )
    sch = ms.tir_integration.compile_tir(database, mod_linear, "llvm --num-cores=1")

    MyModuleWithParams2 = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
    new_func = sch.mod["main"].with_attr("global_symbol", "linear0")
    gv = MyModuleWithParams2.get_global_var("linear0")
    MyModuleWithParams2.update_func(gv, new_func)

    ex = relax.build(MyModuleWithParams2, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithParams2 Prediction:", class_names[pred_kind[0]])
    
    ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=50)
    print("MyModuleWithParams2 time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
