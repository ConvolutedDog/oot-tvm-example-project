import tvm
from tvm.script import tir as T
from tvm.script import relax as R
import tvm.relax as relax

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl


dtype = "float32"

# Load the Dataset
test_data = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.savefig("output.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Class:", class_names[label[0]])


# Download Model Parameters
# wget https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl


# End to End Model Integration
def numpy_mlp(data, w0, b0, w1, b1):
    lv0 = data @ w0.T + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1.T + b1
    return lv2


mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
res = numpy_mlp(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"],
)
print(res)
pred_kind = res.argmax(axis=1)
print(pred_kind)
print("NumPy-MLP Prediction:", class_names[pred_kind[0]])


# Constructing an End to End IRModule in TVMScript
@tvm.script.ir_module
class Network:
    @T.prim_func
    # T.handle creates a TIR var that represents a pointer.
    def relu0(x: T.handle, y: T.handle):
        n = T.int64()
        X = T.match_buffer(param=x, shape=(1, n), dtype=dtype)
        Y = T.match_buffer(param=y, shape=(1, n), dtype=dtype)
        for i, j in T.grid(1, n):
            with T.block("Y"):
                vi, vj = T.axis.remap(kinds="SS", bindings=[i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @T.prim_func
    def linear0(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        """
        |--|         |---------|
        |  |         |         |
        |  | m  X  n |         |
        |  |         |         |
        |--|         |---------|
          1               m
        """
        X = T.match_buffer(param=x, shape=(1, m), dtype=dtype)
        W = T.match_buffer(param=w, shape=(n, m), dtype=dtype)
        B = T.match_buffer(param=b, shape=(n), dtype=dtype)
        Z = T.match_buffer(param=z, shape=(1, n), dtype=dtype)
        Y = T.alloc_buffer(shape=(1, n), dtype=dtype)
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap(kinds="SSR", bindings=[i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap(kinds="SS", bindings=[i, j])
                Z[vi, vj] = B[vj] + Y[vi, vj]

    @R.function
    def main(
        x: R.Tensor((1, "m"), "float32"),
        w0: R.Tensor(("n", "m"), "float32"),
        b0: R.Tensor(("n",), "float32"),
        w1: R.Tensor(("k", "n"), "float32"),
        b1: R.Tensor(("k",), "float32"),
    ):
        m, k, n = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed(
                func="linear0", args=(x, w0, b0), out_sinfo=R.Tensor((1, n), "float32")
            )
            lv1 = R.call_dps_packed(
                func="relu0", args=(lv0), out_sinfo=R.Tensor((1, n), "float32")
            )
            lv2 = R.call_dps_packed(
                func="linear0",
                args=(lv1, w1, b1),
                out_sinfo=R.Tensor((1, k), "float32"),
            )
            R.output(lv2)
        return lv2


mod = Network
mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)


ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

nd_res = vm["main"](
    data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"]
)
print(nd_res)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("Prediction:", class_names[pred_kind[0]])


# Integrate Existing Libraries in the Environment
@tvm.script.ir_module
class NetworkWithExternCall:
    @R.function
    def main(
        x: R.Tensor((1, "m"), "float32"),
        w0: R.Tensor(("n", "m"), "float32"),
        b0: R.Tensor(("n",), "float32"),
        w1: R.Tensor(("k", "n"), "float32"),
        b1: R.Tensor(("k",), "float32"),
    ):
        # block 0
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed(
                "env.linear", (x, w0, b0), R.Tensor((1, n), "float32")
            )
            lv1 = R.call_dps_packed("env.relu", (lv0,), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed(
                "env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32")
            )
            R.output(out)
        return out


@tvm.register_func("env.linear", override=True)
def torch_linear(
    x: tvm.nd.NDArray, w: tvm.nd.NDArray, b: tvm.nd.NDArray, out: tvm.nd.NDArray
):
    # Ref: https://dmlc.github.io/dlpack/latest/python_spec.html
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)


@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray, out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)


mod = NetworkWithExternCall
mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)


ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

nd_res = vm["main"](
    data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"]
)
print(nd_res)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("Prediction:", class_names[pred_kind[0]])


# Mixing TensorIR Code and Libraries
@tvm.script.ir_module
class NetworkMixture:
    @T.prim_func
    def linear0(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n,), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(
        x: R.Tensor((1, "m"), "float32"),
        w0: R.Tensor(("n", "m"), "float32"),
        b0: R.Tensor(("n",), "float32"),
        w1: R.Tensor(("k", "n"), "float32"),
        b1: R.Tensor(("k",), "float32"),
    ):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed(
                "env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32")
            )
            R.output(out)
        return out


mod = NetworkMixture
mod.show(
    black_format=True,
    show_meta=True,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)


ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

nd_res = vm["main"](
    data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"]
)
print(nd_res)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("Prediction:", class_names[pred_kind[0]])


# Bind Parameters to IRModule
NetworkMixtureWithParams = relax.transform.BindParams("main", nd_params)(NetworkMixture)

mod = NetworkMixtureWithParams
mod.show(
    black_format=True,
    show_meta=False,
    verbose_expr=True,
    show_object_address=False,
    show_all_struct_info=True,
)

ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

nd_res = vm["main"](
    data_nd
)
print(nd_res)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("Prediction:", class_names[pred_kind[0]])
