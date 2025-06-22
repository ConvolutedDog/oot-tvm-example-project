import tvm
import tvm.script.relax as R
import tvm.script.tir as T
import numpy as np

import tvm.testing
import tvm.relax.testing.vm


def test_vm_build():
    @tvm.script.ir_module
    class test_vm_build_mod:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            # NOTE `test.vm.identity` is registered in `tvm/relax/testing/vm.py`
            z = R.call_pure_packed("test.vm.identity", x, y,
                                   sinfo_args=(R.Tensor(ndim=2, dtype="float32")))
            return y

    mod = test_vm_build_mod
    target = tvm.target.Target("llvm", host="llvm")
    ex = tvm.relax.build(mod, target, exec_mode="bytecode")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())

    np1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    np2 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))

    vm['foo'](np1, np2)
    tvm.testing.assert_allclose(np2.numpy(), np1.numpy(), rtol=1e-7, atol=1e-7)

    # matmul mod
    @tvm.script.ir_module
    class matmul_mod:
        @R.function
        def matmul(x: R.Tensor((64, 64), "float32"), y: R.Tensor((64, 64), "float32")):
            z = R.matmul(x, y)
            return z

    mod2 = matmul_mod
    target = tvm.target.Target("llvm", host="llvm")
    ex = tvm.relax.build(mod2, target, exec_mode="compiled")

    # BUG @benkangpeng The content printed below is meaningless.
    # ex: VMExecutable
    print(ex.stats())
    print(ex.as_python())
    print(ex.as_text())

    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())

    np1 = np.random.rand(64, 64).astype(np.float32)
    np2 = np.random.rand(64, 64).astype(np.float32)

    np3 = np.matmul(np1, np2)
    res = vm['matmul'](tvm.nd.array(np1), tvm.nd.array(np2))
    tvm.testing.assert_allclose(res.numpy(), np3, rtol=1e-5, atol=1e-5)
    # print(res.numpy())


def test_vmcodegen():
    @tvm.script.ir_module
    class test_vmcodegen_mod:
        @T.prim_func
        def matmul(x: T.Buffer((16, 32), "float32"), y: T.Buffer((32, 64), "float32"), z: T.Buffer((16, 64), "float32")):
            T.func_attr({"global_symbol": "matmul"})
            for i, j, k in T.grid(16, 64, 32):
                with T.block("T_matmul"):
                    i_1, j_1, k_1 = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        z[i_1, j_1] = T.float32(0)
                    z[i_1, j_1] = z[i_1, j_1] + x[i_1, k_1] * y[k_1, j_1]

    builder = tvm.relax.ExecBuilder()
    mod = tvm.relax.vm_build._vmcodegen(
        builder, test_vmcodegen_mod, exec_mode="compiled")
    print(mod)


if __name__ == "__main__":
    test_vm_build()
    test_vmcodegen()
