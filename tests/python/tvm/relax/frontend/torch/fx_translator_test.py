# Convert a Pytorch FX GraphModule to a Relax Module

import torch
from tvm.relax.frontend.torch import from_fx


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=10, out_features=7, bias=True)

    def forward(self, input):
        return self.linear(input)


# Instantiate the model and create the input info dict.
torch_model = MyModule()
input_info = [((128, 10), torch.float32)]
input_tensors = [
    torch.randn(*shape, dtype=dtype)
    for shape, dtype in input_info
]

# Use FX tracer to trace the PyTorch model.
graph_module = torch.fx.symbolic_trace(torch_model)

mod = from_fx(graph_module, input_info)
mod.show()
