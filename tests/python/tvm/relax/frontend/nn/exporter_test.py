# exporter.py is the helper file used by function `export_tvm`
from tvm.relax.frontend import nn
import tvm.relax as rx
import tvm
import numpy as np


def test_builtin_module():
    mod = nn.modules.ReLU()
    export_mod, _ = mod.export_tvm(
        {"forward": {"x": nn.spec.Tensor((4, 4), "float32")}}
    )

    print(export_mod)


def test_custom_module():
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
    tvm_mod, param_spec = mod.export_tvm(
        {"forward": {"x": nn.spec.Tensor((1, 10), "float32")}}
    )
    tvm_mod.show()
    print(param_spec)

    ex = rx.build(tvm_mod, tvm.target.Target("llvm", "llvm"))
    vm = rx.VirtualMachine(ex, tvm.cpu())

    in_data = tvm.nd.array(np.random.rand(1, 10).astype("float32"))
    params = [np.random.rand(*param.shape).astype("float32")
              for _, param in param_spec]
    params = [tvm.nd.array(param) for param in params]

    print(vm["forward"](in_data, *params).numpy())


def test_dynamic_shape():
    linear = nn.Linear(128, 128, bias=False)

    exported_mod, param_spec = linear.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor(("batch_size", 128), "float32")}}
    )

    exported_mod.show()

def test_export_module_with_debug():
    class MyModule(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.linear_relu_stack = nn.ModuleList(
                [nn.Linear(self.in_features, self.out_features, bias=False), nn.ReLU()])
        
        def forward(self, x: nn.Tensor):
            return self.linear_relu_stack(x)
        
    mod = MyModule(10, 10)
    exported_mod, param_spec = mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((1, 10), "float32")}},
        debug=True,
    )
    exported_mod.show()
    print(param_spec)

if __name__ == "__main__":
    test_builtin_module()
    test_custom_module()
    test_dynamic_shape()
    test_export_module_with_debug()
