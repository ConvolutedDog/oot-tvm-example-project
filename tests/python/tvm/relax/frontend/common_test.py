import tvm
import tvm.script.relax as R
from tvm.relax.frontend import detach_params

def test_detach_params():
    param1 = tvm.nd.empty((2,), "float32")
    param2 = tvm.nd.empty((3,), "float32")

    @tvm.script.ir_module
    class Module:
        @R.function
        def func(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"params": [param1, param2]})
            return x

    mod = Module
    detached_mod, detached_params = detach_params(mod)
    print(detached_mod)
    print(detached_params)

if __name__ == "__main__":
    test_detach_params()