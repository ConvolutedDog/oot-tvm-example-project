# Import the exported Pytorch model(torch.export.ExportedProgram).
# Convert the exported Pytorch model to a Relax model.


import torch
import tvm
from tvm.relax.frontend.torch import from_exported_program

def test_exported_program_translator():
    class torchModel(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features, bias=False)
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            res = self.linear(x)
            res = self.relu(res)
            return res
    

    torch_model = torchModel(10, 12)
    x = torch.rand([1,10], dtype=torch.float32)

    # Use `torch.export.export()` to convert the Pytorch Model into `Pytorch ExportedProgram`.
    exported_program = torch.export.export(torch_model,args=(x,))
    # print(exported_program)

    relax_model = from_exported_program(exported_program)

    relax_model.show()

if __name__ == '__main__':
    test_exported_program_translator() 