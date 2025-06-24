import tvm
from tvm.relax.frontend import nn

# NOTE nn.Module is a subclass of nn.SubroutineMixin.
# Set the `define_subroutine` attribute to True to enable subroutine dispatch.
# Enable subroutine dispatch means that when forward is called, instead of inlining
# the computation, it creates a separate function in the IR that can be reused, optimized,
# and makes the IR more modular and readable.


def test_subroutine():
    class Model(nn.Module):
        define_subroutine = True

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weights = nn.Parameter(
                (in_features, out_features), dtype="float32")

        def forward(self, x: nn.Tensor):
            # A sperate function will be created to implement the matrix multiplication.
            y = nn.op.matmul(x, self.weights)
            z = nn.op.relu(y)
            return z

    m = Model(8, 16)
    irmodule, _ = m.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor([1, 8], "float32")}}
    )

    irmodule.show()


if __name__ == "__main__":
    test_subroutine()
