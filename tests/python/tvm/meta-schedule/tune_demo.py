import tvm
import torch
import tvm.script
import tvm.script.tir as T
import tvm.script.relax as R
from tvm.relax.frontend.torch import from_exported_program
import tvm.meta_schedule as ms
import tvm.relax as rx
import tvm.testing

import numpy as np


def test_tune_tir_module_exported_from_torch():
    # Create a simple PyTorch model
    class torchModule(torch.nn.Module):
        def __init__(self, in_features, out_features) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    my_model = torchModule(10, 10)
    x = torch.rand([1, 10], dtype=torch.float32)

    # Export the PyTorch model to TVM
    exported_program = torch.export.export(my_model, args=(x,))
    ir_mod = from_exported_program(exported_program)

    print("Original Relax Module:")
    ir_mod.show()

    # Lower to TIR
    from tvm.relax.pipeline import zero_pipeline
    tir_mod = zero_pipeline()(ir_mod)

    print("\nLowered TIR Module:")
    tir_mod.show()

    # Extract the TIR function for tuning
    # The issue was that tune_tir expects a module with a single TIR function
    # Extract the fused function that we want to tune
    prim_func_name = None
    for gv, func in tir_mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc) and "fused" in gv.name_hint:
            prim_func_name = gv.name_hint
            break

    if prim_func_name:
        print(f"\nExtracting function: {prim_func_name}")
        # Create a new module with just this function
        extracted_mod = tvm.IRModule()
        extracted_mod[prim_func_name] = tir_mod[prim_func_name]

        # Tune the extracted TIR function
        database = ms.tune_tir(
            mod=extracted_mod,
            target="llvm --num-cores=1",
            max_trials_global=16,
            num_trials_per_iter=16,
            work_dir="./tune_tmp",
        )

        # Compile the tuned function - specify the workload name to match the function name
        sch = ms.tir_integration.compile_tir(
            database, extracted_mod, "llvm --num-cores=1", workload_name=prim_func_name)

        print("\nTuned Schedule:")
        if sch is not None:
            print(sch)
            sch.mod.show()
        else:
            print("No schedule found in the database")
    else:
        print("No suitable TIR function found for tuning")


def test_tune_relax_module_exported_from_torch():
    # Create a simple PyTorch model
    class torchModule(torch.nn.Module):
        def __init__(self, in_features, out_features) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    my_model = torchModule(10, 10)
    x = torch.rand([1, 10], dtype=torch.float32)
    exported_program = torch.export.export(my_model, args=(x,))
    ir_mod = from_exported_program(exported_program)
    print("Original Relax Module:")
    ir_mod.show()

    params = {}
    database = ms.relax_integration.tune_relax(
        mod=ir_mod,
        params=params,
        target="llvm --num-cores=1",
        max_trials_global=16,
        num_trials_per_iter=16,
        work_dir="./tune_tmp",
    )

    sch = ms.relax_integration.compile_relax(
        database, ir_mod, "llvm --num-cores=1", params=params)
    print("\nTuned Schedule:")
    print(sch)
    sch.mod.show()


def test_tune():
    @tvm.script.ir_module
    class MyTirModule:
        @T.prim_func
        def matmul(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):  # type: ignore
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i, j, k in T.grid(128, 128, 128):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

    database = ms.tune_tir(
        mod=MyTirModule,
        target="llvm --num-cores=1",
        max_trials_global=16,
        num_trials_per_iter=16,
        work_dir="./tune_tmp",
    )
    sch = ms.tir_integration.compile_tir(
        database, MyTirModule, "llvm --num-cores=1")
    sch.mod.show()

    a_nd = tvm.nd.array(np.random.rand(128, 128).astype("float32"))
    b_nd = tvm.nd.array(np.random.rand(128, 128).astype("float32"))
    c_nd = tvm.nd.array(np.zeros((128, 128), dtype="float32"))
    c_nd_2 = tvm.nd.array(np.zeros((128, 128), dtype="float32"))

    lib = tvm.build(MyTirModule, target="llvm")
    f_timer_before = lib.time_evaluator("main", tvm.cuda(4))
    print("Time cost of MyModule before tuning: %.3f ms" %
          (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))

    lib = tvm.build(sch.mod, target="llvm")
    f_timer_after = lib.time_evaluator("main", tvm.cuda(4))
    print("Time cost of MyModule after tuning: %.3f ms" %
          (f_timer_after(a_nd, b_nd, c_nd_2).mean * 1000))

    tvm.testing.assert_allclose(
        c_nd.numpy(), c_nd_2.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    # test_tune_tir_module_exported_from_torch() #BUG
    # test_tune_relax_module_exported_from_torch() #BUG
    test_tune()
