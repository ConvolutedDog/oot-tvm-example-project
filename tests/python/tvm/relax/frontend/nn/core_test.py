import tvm
from tvm.relax.frontend.nn.core import Module, wrap_nested
import tvm.script.relax as R
import tvm.relax as rx
from tvm.relax.frontend.nn.core import Tensor
from tvm.relax.frontend.nn import op
import numpy as np

def test_Module():
    class MyModule(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor):
            return op.matmul(x, x)

    npi = np.random.randn(16,16).astype("float32")
    x = Tensor.from_const(npi)

    # BUG @benkangpeng
    mod = MyModule()
    y = mod(x)
    print(y)


def test_wrap_nested():
    pass
    # pv0 = rx.Tuple([1, 2, 3])
    # BUG Don't know how to use wrap_nested
    # print(wrap_nested(pv0, "tensor0"))


if __name__ == "__main__":
    test_Module()
    test_wrap_nested()
