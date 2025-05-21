import tvm.relax as rx
from tvm import tir
from tvm.script import ir as I, relax as R, tir as T


m = tir.Var("m", "int64")
n = tir.Var("n", "int64")
x = rx.Var("x", R.Tensor([m, n], "float32"))
gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))

call_node = rx.op.add(x, gv0)

_bindings = [rx.VarBinding(gv1, call_node)]

_blocks = [rx.BindingBlock(_bindings)]

_seq_expr = rx.SeqExpr(_blocks, gv1)

call_node.show()

_bindings[0].show()

_blocks[0].show()

_seq_expr.show()

cond = rx.Var("condition", R.Tensor([], "bool"))

v_in_if = rx.Var("v_in_if", R.Tensor([m, n], "float32"))
gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))

# build the true branch
true_bindings = [
    rx.VarBinding(v_in_if, rx.op.add(x, x)),
    rx.VarBinding(gv0, rx.op.multiply(x, x)),
]

true_blocks = [rx.BindingBlock(true_bindings)]

true_seq_expr = rx.SeqExpr(true_blocks, true_blocks[-1].bindings[-1].var)

# build the false branch
false_bindings = [
    rx.VarBinding(v_in_if, rx.op.multiply(x, x)),
    rx.VarBinding(gv0, rx.op.add(x, x)),
]
false_blocks = [rx.BindingBlock(false_bindings)]
false_seq_expr = rx.SeqExpr(false_blocks, false_blocks[-1].bindings[-1].var)

# build If node
if_node = rx.If(cond=cond, true_branch=true_seq_expr, false_branch=false_seq_expr)

if_node.show()

# Function

scalar_struct_info = rx.TensorStructInfo(shape=[], dtype="int32")
gv0 = rx.Var("gv0", scalar_struct_info)

f = rx.Var("f", rx.FuncStructInfo([scalar_struct_info], scalar_struct_info))

ipt = rx.Var("ipt", scalar_struct_info)
x0 = rx.Var("x0", scalar_struct_info)
x1 = rx.Var("x1", scalar_struct_info)
x2 = rx.Var("x2", scalar_struct_info)
y = rx.Var("y", scalar_struct_info)


inner_block = rx.BindingBlock(
    [rx.VarBinding(x0, rx.const(2, "int32")), rx.VarBinding(y, rx.Call(f, [x0]))]
)

inner_func = rx.Function([ipt], rx.SeqExpr([inner_block], y), scalar_struct_info)

outer_block = rx.BindingBlock(
    [
        rx.VarBinding(f, inner_func),
        rx.VarBinding(x1, rx.const(1, "int32")),
        rx.VarBinding(x2, rx.op.add(x1, rx.Call(f, [x1]))),
        rx.VarBinding(gv0, x2),
    ]
)
func = rx.Function([], rx.SeqExpr([outer_block], gv0), scalar_struct_info)
func.show()
