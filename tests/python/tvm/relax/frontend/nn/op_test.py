from tvm.script import relax as R
from tvm.script import tir as T
import tvm.relax as rx
from tvm.relax.frontend import nn
import tvm
import numpy as np

func_map = {}


def register(func):
    func_map[func.__name__] = func
    return func


@register
def test_manipulate():
    class MyModel(nn.Module):
        def test(self, x: nn.Tensor):
            # unsqueeze
            z0 = nn.op.unsqueeze(x, dim=0)  # R.Tensor(1,2,10)
            z1 = nn.op.unsqueeze(x, dim=1)  # R.Tensor(2,1,10)
            z2 = nn.op.unsqueeze(x, dim=2)  # R.Tensor(2,10,1)

            # concat
            z3 = nn.op.concat([x, x], dim=0)  # R.Tensor(4,10)
            z4 = nn.op.concat([x, x], dim=1)  # R.Tensor(2,20)

            return (z0, z1, z2, z3, z4)

    m = MyModel()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": nn.spec.Tensor([2, 10], "float32")}}
    )
    irmodule.show()


@register
def test_binary():
    class MyModel(nn.Module):
        def test(self, x: nn.Tensor, y: nn.Tensor):
            # element-wise
            z0 = nn.op.add(x, y)
            z1 = nn.op.subtract(x, y)
            z2 = nn.op.multiply(x, y)
            z3 = nn.op.divide(x, y)

            # matrix multiplication
            z4 = nn.op.matmul(x, y)
            return (z0, z1, z2, z3, z4)

    model = MyModel()
    irmodule, _ = model.export_tvm(
        spec={"test": {"x": nn.spec.Tensor(
            [2, 2], "float32"), "y": nn.spec.Tensor([2, 2], "float32")}}
    )
    irmodule.show()
    ex = rx.build(irmodule, target="llvm")
    vm = rx.VirtualMachine(ex, tvm.cpu())
    x = tvm.nd.array(np.array([[1, 2], [3, 4]], dtype="float32"))
    y = tvm.nd.array(np.array([[5, 6], [7, 8]], dtype="float32"))
    res = vm["test"](x, y)

    for elem in res:
        print(elem.numpy())


@register
def test_chunk():
    class MyModel(nn.Module):
        def test(self, x: nn.Tensor):
            # Split a tensor along dim into the specified number of chunks
            # R.Tuple(R.Tensor(6,3), R.Tensor(6,3))
            z0 = nn.op.chunk(x, chunks=2, dim=1)
            # R.Tuple(R.Tensor(2,6), R.Tensor(2,6), R.Tensor(2,6))
            z1 = nn.op.chunk(x, chunks=3, dim=0)
            return (z0, z1)

    model = MyModel()
    irmodule, _ = model.export_tvm(
        spec={"test": {"x": nn.spec.Tensor([6, 6], "float32")}}
    )
    irmodule.show()


@register
def test_sum_max_min():
    class MyModel(nn.Module):
        def test(self, x: nn.Tensor):
            # axis=0: sum over rows(vertical)
            z0 = nn.op.sum(x, axis=0)  # R.Tensor((6,))
            # axis=1: sum over columns(horizontal)
            z1 = nn.op.sum(x, axis=1)  # R.Tensor((4,))

            z2 = nn.op.sum(x, axis=0, keepdims=True)  # R.Tensor((1,6))
            z3 = nn.op.sum(x, axis=-1, keepdims=True)  # R.Tensor((4,1))

            z4 = nn.op.max(x, axis=0)  # R.Tensor((6,))
            z5 = nn.op.min(x, axis=1)  # R.Tensor((4,))

            return (z0, z1, z2, z3, z4, z5)

    mod = MyModel()
    irmodule, _ = mod.export_tvm(
        spec={"test": {"x": nn.spec.Tensor([4, 6], "float32")}}
    )
    irmodule.show()

    ex = rx.build(irmodule, target="llvm")
    vm = rx.VirtualMachine(ex, tvm.cpu())
    x = tvm.nd.array(np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [
                     13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]], dtype="float32"))
    res = vm["test"](x)
    for elem in res:
        print(elem.numpy())


@register
def test_matmul():
    class MyModel(nn.Module):
        def test(self, x: nn.Tensor, y: nn.Tensor):
            z0 = nn.op.matmul(x, y)#R.Tensor(15, 3, 6)
            return z0

    model = MyModel()
    
    #[5, 6] broadcast-> [1, 5, 6]
    # broadcast ref: https://data-apis.org/array-api/latest/API_specification/broadcasting.html#broadcasting
    irmodule, _ = model.export_tvm(
        spec={"test": {"x": nn.spec.Tensor(
            [15, 3, 5], "float32"), "y": nn.spec.Tensor([5, 6], "float32")}}
    )
    irmodule.show()


if __name__ == "__main__":
    for func in func_map.values():
        print(f'⭕⭕⭕{func.__name__} is running...')
        func()
        print(f'✅✅✅{func.__name__} is done!')
