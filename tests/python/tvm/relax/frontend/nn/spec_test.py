from tvm.relax.frontend import nn


def test_MethodSpec():

    def myFunc(x: nn.Tensor, y: nn.Tensor) -> nn.Tensor:
        return nn.op.add(x, y)

    # `MethodSpec` is a class that defines/specifies the signature of a method.
    # NOTE We must use the type in module `spec`(e.g. `nn.spec.Tensor`,`nn.spec.Int`) to define the type of the arguments in MethodSpec.
    spec0 = nn.spec.MethodSpec(myFunc, ["x", "y"], [nn.spec.Tensor(
        ["m", "n"], "float32"), nn.spec.Tensor(["m", "n"], "float32")], "plain", "plain")

    print(spec0)

    spec1 = nn.spec.MethodSpec.from_raw(
        spec={"x": nn.spec.Tensor(["m", "n"], "float32"),
              "y": nn.spec.Tensor(["m", "n"], "float32")},
        method=myFunc
    )

    print(spec1)

    # Get the method spec from a torch function.
    import torch

    def myFuncTorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    spec2 = nn.spec.MethodSpec.from_torch(
        args=[torch.empty([16, 16], dtype=torch.float32),
              torch.empty([16, 16], dtype=torch.float32)],
        method=myFuncTorch
    )

    print(spec2)


def test_ModuleSpec():
    class MyModule(nn.Module):
        def __init__(self, in_features, out_features):
            self.weights = nn.Parameter([in_features, out_features], "float32")

        def forward(self, x: nn.Tensor):
            y = nn.op.matmul(x, self.weights)
            y = nn.op.relu(y)
            return y

    method_spec0 = nn.spec.MethodSpec.from_raw(
        spec={"x": nn.spec.Tensor(["m", "n"], "float32")},
        method=MyModule.forward
    )

    spec0 = nn.spec.ModuleSpec.from_raw(
        spec={"forward": {"x": nn.spec.Tensor(["m", "n"], "float32")}},
        module=MyModule
    )

    print(spec0)


if __name__ == "__main__":
    test_MethodSpec()
    test_ModuleSpec()
