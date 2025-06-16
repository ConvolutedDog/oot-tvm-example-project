import tvm
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax import ExecBuilder
import numpy as np

@tvm.script.ir_module
class Model:
    @R.function
    def model(x: R.Tensor([1, 3, 224, 224], "float32")):
        weight = R.const(np.random.randn(16, 3, 3, 3), "float32")
        conv = tvm.relax.op.nn.conv2d(x, weight)
        return conv

# Build the model
target = tvm.target.Target("llvm")
ex = tvm.relax.build(Model, target)

eb = ExecBuilder()
eb.declare_function("test.vm.add", 1)
with eb.function("func0", num_inputs=2):
    eb.emit_call("test.vm.add", args=[eb.r(0), eb.r(1)], dst=eb.r(2))
    eb.emit_ret(eb.r(2))

ex = eb.get()
print(ex)