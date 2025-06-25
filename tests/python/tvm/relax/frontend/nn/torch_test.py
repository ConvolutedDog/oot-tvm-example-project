import tvm
import tvm.relax as rx
import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn.torch import TorchModule

import numpy as np
import torch


class MyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_relu_stack = nn.ModuleList(
            [nn.Linear(self.in_features, 512, bias=False), nn.ReLU(),
             nn.Linear(512, self.out_features, bias=False), nn.ReLU()])

    def forward(self, x: nn.Tensor):
        return self.linear_relu_stack(x)


mod = MyModule(10, 10)


irmodule, param_spec = mod.export_tvm(
    spec={"forward": {"x": nn.spec.Tensor([1, 10], "float32")}},
    debug=False
)

ex = rx.build(irmodule, target="llvm")
vm = rx.VirtualMachine(ex, tvm.cpu())

print(irmodule)
print(param_spec)

# _x = tvm.nd.array(np.random.randn(1, 10).astype("float32"))
_x = torch.randn(1, 10, dtype=torch.float32)
params = [tvm.nd.array(np.random.randn(*param.shape).astype("float32"))
          for _, param in param_spec]


spec0 = nn.spec.ModuleSpec.from_raw(
    spec={"forward": {"x": nn.spec.Tensor([1, 10], "float32")}},
    module=mod
)
MyTorchModule = TorchModule(spec0, vm, params)

res = MyTorchModule["forward"](_x)
print(res)
