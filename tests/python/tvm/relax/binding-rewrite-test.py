import tvm
from tvm.relax.binding_rewrite import DataflowBlockRewrite
from tvm.relax.analysis import name_to_binding
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.script.ir_module
class Identity:
    @R.function
    def main(x: R.Tensor([32, 32], "int32")) -> R.Tensor:
        with R.dataflow():
            lv0 = R.add(x, x)
            R.output(lv0)
        return lv0


# Identity.show()

root_fn = Identity["main"]
dfb = root_fn.body.blocks[0]

dfb.show()

rewrite = DataflowBlockRewrite(dfb, root_fn)
# `is_dfvar = True` means that the variable is a dataflow variable，which can only be access in the scope of the dataflow
rewrite.add(Identity["main"].params[0], "unused", is_dfvar=True)
rewrite.mutated_root_fn().show()


@tvm.script.ir_module
class IdentityUnused:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            unused = lv0
            R.output(lv0)
        return lv0

n2binding = name_to_binding(IdentityUnused["main"])
rewrite.remove_unused(n2binding["unused"][0].var)
rewrite.mutated_root_fn().show()