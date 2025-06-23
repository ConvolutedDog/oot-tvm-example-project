import tvm
import tvm.script.relax as R
import tvm.relax as rx
from tvm.relax.frontend import nn
import numpy as np

func_dict = {}


def register(func):
    func_dict[func.__name__] = func
    return func


@register
def test_Parameter():
    param0 = nn.Parameter((4, 4), "float32")
    print(param0)
    print(param0.data)
    param0.data = np.random.randn(4, 4).astype("float32")
    print(param0.data)

    param1 = nn.Parameter((4, 4), "float32")
    # change the dtype of param1
    param1.to("int32")
    print(param1.dtype)


@register
def test_Module():
    # define a custom model
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x

    mod, param_spec = MyModel().export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
    )
    mod.show()
    print(param_spec)
    # [('fc1.weight', Tensor([256, 784], "float32")), ('fc1.bias', Tensor([256], "float32")), 
    # ('fc2.weight', Tensor([10, 256], "float32")), ('fc2.bias', Tensor([10], "float32"))]

    mod = rx.get_pipeline("zero")(mod)
    target = tvm.target.Target("llvm", host="llvm")
    ex = rx.build(mod, target=target)
    vm = rx.VirtualMachine(ex, tvm.cpu())

    _data = np.random.randn(1, 784).astype("float32")
    tvm_data = tvm.nd.array(_data, device=tvm.cpu())

    # generate the random values for the weights and biases
    params = [np.random.rand(*param.shape).astype("float32")
              for _, param in param_spec]
    params = [tvm.nd.array(param, device=tvm.cpu()) for param in params]
    print(vm["forward"](tvm_data, *params).numpy())


@register
def test_module_list():
    class MyModule(nn.Module):
        def __init__(self):
            self.layers = nn.ModuleList(
                [nn.Linear(4, 4, bias=False), nn.ReLU()])

        def forward(self, x: nn.Tensor):
            return self.layers(x)

    mod = MyModule()
    print(dict(mod.named_parameters()))
    print(mod.state_dict())


@register
def test_wrap_nested():
    pass
    # pv0 = rx.Tuple([1, 2, 3])
    # BUG Don't know how to use wrap_nested
    # print(wrap_nested(pv0, "tensor0"))


if __name__ == "__main__":
    for func in func_dict.values():
        print(f"⭕⭕⭕Running {func.__name__}")
        func()
        print(f"✅✅✅Running {func.__name__} done")
