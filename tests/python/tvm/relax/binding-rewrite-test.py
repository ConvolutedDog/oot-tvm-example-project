import tvm
from tvm.relax.binding_rewrite import DataflowBlockRewrite
from tvm.relax.analysis import name_to_binding
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax import Binding


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

# Due to the immutable and copy-on-write nature of TVM AST nodes, the rewriting is not done in
# place. Instead, a new DataflowBlock is created and returned with mutated_dfb. Similarly, its new
# root Function is created and returned by mutated_root_fn. To apply this change for an IRModule,
# use mutate_irmodule which rewrites the old function that registered in the constructor.
rewrite.mutated_root_fn().show()
rewrite.mutate_irmodule(Identity).show()


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
print(n2binding)
root_fn = IdentityUnused["main"]
dfb = root_fn.body.blocks[0]
rewrite = DataflowBlockRewrite(dfb, root_fn)

rewrite.remove_unused(n2binding["unused"][0].var)
rewrite.mutated_root_fn().show()
rewrite.mutate_irmodule(IdentityUnused).show()


@tvm.script.ir_module
class Lv0To1:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
        #   lv0 => lv1
        #  /   \
        # lv2  lv3
        #  \   /
        #   lv4
        with R.dataflow():
            lv0: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                "my_relu", (x,), R.Tensor((32, 32), dtype="float32")
            )
            lv1: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                "my_sigmoid", (x,), R.Tensor((32, 32), dtype="float32")
            )
            lv2: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                "my_add", (x, lv0), R.Tensor((32, 32), dtype="float32")
            )
            lv3: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                "my_mul", (x, lv0), R.Tensor((32, 32), dtype="float32")
            )
            lv4: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                "my_whatever", (lv2, lv3), R.Tensor(
                    (32, 32), dtype="float32")
            )
            R.output(lv4)
        return lv4

        #   lv0 => lv1
        #  /   \
        # lv2  lv3
        #  \   /
        #   lv4


root_fn = Lv0To1["main"]
dfb = root_fn.body.blocks[0]
rewrite = DataflowBlockRewrite(dfb, root_fn)

n2binding = name_to_binding(Lv0To1["main"])
lv0 = n2binding["lv0"][0].var
lv1 = n2binding["lv1"][0].var

rewrite.replace_all_uses(lv0, lv1)

rewrite.remove_all_unused()
rewrite.mutated_root_fn().show()
rewrite.mutate_irmodule(Lv0To1).show()
        #   lv1 => lv0
        #  /   \
        # lv2  lv3
        #  \   /
        #   lv4
