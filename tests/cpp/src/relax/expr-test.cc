#include "relax/expr-test.h"
#include "dlpack/dlpack.h"
#include "tvm/relax/struct_info.h"
#include <tvm/node/script_printer.h>
#include <tvm/relax/expr.h>

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace expr_test {

void CallTest() {
  LOG_SPLIT_LINE("CallTest");

  /// Create Expr
  Expr opexpr = tvm::Op::Get("relax.nn.conv2d");
  LOG_PRINT_VAR(opexpr);

  Expr globalvarexpr = GlobalVar("testglobalvar");
  LOG_PRINT_VAR(globalvarexpr);

  /// Create tvm::relax::Function
  Var arg1{"arg1", tvm::relax::ShapeStructInfo{4}};
  Var arg2{"arg2", tvm::relax::ShapeStructInfo{4}};
  Call call{
      opexpr, {arg1, arg2}
  };

  Call callwithfields =
      WithFields(call, tvm::Op::Get("relax.argmax"),
                 tvm::runtime::Optional<tvm::runtime::Array<Expr>>{{globalvarexpr}});
  LOG_PRINT_VAR(callwithfields);

  Function func{
      {arg1, arg2},
      call,
      tvm::relax::ShapeStructInfo{4},
      true,
  };

  LOG_PRINT_VAR(func);
}

void TupleTest() {
  LOG_SPLIT_LINE("TupleTest");

  /// Create Expr
  Expr opexpr = tvm::Op::Get("relax.nn.conv2d");
  LOG_PRINT_VAR(opexpr);

  Expr globalvarexpr = GlobalVar("testglobalvar");
  LOG_PRINT_VAR(globalvarexpr);

  Tuple tuple{
      {
       opexpr, }
  };
  LOG_PRINT_VAR(tuple);

  Tuple tuplewithfields =
      WithFields(tuple, tvm::runtime::Optional<tvm::runtime::Array<Expr>>{
                            {opexpr, globalvarexpr}
  });
  LOG_PRINT_VAR(tuplewithfields);
}

void TupleGetItemTest() {
  LOG_SPLIT_LINE("TupleGetItemTest");

  Var arg1{"arg1", tvm::relax::ShapeStructInfo{4}};
  Var arg2{"arg2", tvm::relax::ShapeStructInfo{4}};
  Tuple tuple{
      {arg1, arg2}
  };
  TupleGetItem tuplegetitem{tuple, 1};
  LOG_PRINT_VAR(tuplegetitem);

  Expr globalvarexpr = GlobalVar("testglobalvar");
  TupleGetItem tuplegetitemwithfields =
      WithFields(tuplegetitem, {globalvarexpr}, tvm::Integer{0});
  LOG_PRINT_VAR(tuplegetitemwithfields);
}

void LeafExprTest() {
  LOG_SPLIT_LINE("LeafExprTest");

  /// ShapeExpr
  ShapeExpr shapeexpr{
      {1, 2, 3, 4}
  };
  LOG_PRINT_VAR(shapeexpr);

  /// Var
  Var var{"testvar", tvm::relax::ShapeStructInfo{4}};
  LOG_PRINT_VAR(var);
  LOG_PRINT_VAR(var->vid->name_hint);

  /// DataflowVar
  DataflowVar dataflowvar{"testdataflowvar", tvm::relax::ShapeStructInfo{4}};
  LOG_PRINT_VAR(dataflowvar);

  /// Constant
  NDArray ndarrayconstvalue2 =
      NDArray::Empty(ShapeTuple{1, 2, 3, 4}, DLDataType{DLDataTypeCode::kDLFloat, 32, 1},
                     DLDevice{DLDeviceType::kDLCPU, 0});
  LOG_PRINT_VAR(ndarrayconstvalue2);
  Constant constant2{ndarrayconstvalue2};
  LOG_PRINT_VAR(constant2);

  /// PrimValue
  PrimValue primvalue{1};
  LOG_PRINT_VAR(primvalue);
  PrimValue primvalue2{
      tvm::IntImm{DataType{DLDataType{DLDataTypeCode::kDLInt, 32, 1}}, 2}
  };
  LOG_PRINT_VAR(primvalue2);

  /// StringImm
  StringImm stringimm{"hello"};
  LOG_PRINT_VAR(stringimm);

  /// DataTypeImm
  DataTypeImm datatypeimm{DataType{DLDataType{DLDataTypeCode::kDLInt, 32, 1}}};
  LOG_PRINT_VAR(datatypeimm);
}

void BindTest() {
  LOG_SPLIT_LINE("BindTest");

  /// Binding
  /// `BindingNode` is the base class of a variable binding in Relax.

  /// MatchCast
  ///
  /// @code{.python}
  /// import tvm.relax as rx
  /// from tvm import tir
  /// from tvm.script import ir as I, relax as R, tir as T
  ///
  /// m = tir.Var("m", dtype="int64")
  /// n = tir.Var("n", dtype="int64")
  /// shape = rx.const([16, 8], "int32")
  /// var = rx.Var("v0", R.Shape())
  /// b0 = rx.MatchCast(var, shape, R.Tensor([m, n], "int32"))
  ///
  /// b0.show()
  /// # Output:
  /// # m = T.int64()
  /// # n = T.int64()
  /// # v0: R.Shape(ndim=-1) = R.match_cast(
  /// #     metadata["relax.expr.Constant"][0],
  /// #     R.Tensor((m, n), dtype="int32")
  /// # )
  /// @endcode
  {
    tvm::tir::Var m{"m", DataType::Int(64)};
    tvm::tir::Var n{"n", DataType::Int(64)};
    Var var{"v0", ShapeStructInfo{-1}};
    NDArray ndarrayconstvalue2 =
        NDArray::Empty(ShapeTuple{16, 8}, DLDataType{DLDataTypeCode::kDLInt, 64, 1},
                       DLDevice{DLDeviceType::kDLCPU, 0});
    Constant shape{ndarrayconstvalue2};
    MatchCast b0{
        var, shape, TensorStructInfo{ShapeExpr{{m, n}}, DataType::Int(32)}
    };
    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(b0, tvm::PrinterConfig{}));
    /// Output:
    /// m = T.int64()
    /// n = T.int64()
    /// v0: R.Shape(ndim=-1) = R.match_cast(
    ///     metadata["relax.expr.Constant"][0],
    ///     R.Tensor((m, n), dtype="int32")
    /// )
  }

  /// VarBinding & BindingBlock & SeqExpr
  ///
  /// @code{.python}
  /// import tvm.relax as rx
  /// from tvm import tir
  /// from tvm.script import ir as I, relax as R, tir as T
  ///
  /// m = tir.Var("m", "int64")
  /// n = tir.Var("n", "int64")
  /// x = rx.Var("x", R.Tensor([m, n], "float32"))
  ///
  /// gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
  /// gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
  /// call_node = rx.op.add(x, gv0)
  /// _bindings = [rx.VarBinding(gv1, call_node)]
  /// _blocks = [rx.BindingBlock(_bindings)]
  /// _seq_expr = rx.SeqExpr(_blocks, gv1)
  ///
  /// call_node.show()
  /// # Output:
  /// # R.add(x, gv0)
  ///
  /// _bindings[0].show()
  /// # Output:
  /// # m = T.int64()
  /// # n = T.int64()
  /// # x: R.Tensor((m, n), dtype="float32")
  /// # gv0: R.Tensor((m, n), dtype="float32")
  /// # gv1: R.Tensor((m, n), dtype="float32") = R.add(x, gv0)
  ///
  /// _blocks[0].show()
  /// # Output:
  /// # m = T.int64()
  /// # n = T.int64()
  /// # x: R.Tensor((m, n), dtype="float32")
  /// # gv0: R.Tensor((m, n), dtype="float32")
  /// # gv1: R.Tensor((m, n), dtype="float32") = R.add(x, gv0)
  ///
  /// _seq_expr.show()
  /// # Output:
  /// # m = T.int64()
  /// # n = T.int64()
  /// # x: R.Tensor((m, n), dtype="float32")
  /// # gv0: R.Tensor((m, n), dtype="float32")
  /// # gv1: R.Tensor((m, n), dtype="float32") = R.add(x, gv0)
  /// # gv1
  /// @endcode
  {
    tvm::tir::Var m{"m", DataType::Int(64)};
    tvm::tir::Var n{"n", DataType::Int(64)};
    Var x{
        "x", TensorStructInfo{ShapeExpr{{m, n}}, DataType::Float(32)}
    };
    Var gv0{
        "gv0", TensorStructInfo{ShapeExpr{{m, n}}, DataType::Float(32)}
    };
    Var gv1{
        "gv1", TensorStructInfo{ShapeExpr{{m, n}}, DataType::Float(32)}
    };

    OpRegistry *opregistry = OpRegistry::Global();
    // LOG_PRINT_VAR(opregistry->ListAllNames());
    // NOLINTNEXTLINE
    Call call_node{
        Op::Get("relax.add"), {x, gv0}
    };

    // NOLINTNEXTLINE
    tvm::Array<Binding> _bindings{{VarBinding(gv1, call_node)}};
    // NOLINTNEXTLINE
    tvm::Array<BindingBlock> _blocks{{BindingBlock(_bindings)}};
    // NOLINTNEXTLINE
    SeqExpr seq_expr(_blocks, gv1);

    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(_bindings[0], tvm::PrinterConfig{}));
    /// Output:
    /// m = T.int64()
    /// n = T.int64()
    /// x: R.Tensor((m, n), dtype="float32")
    /// gv0: R.Tensor((m, n), dtype="float32")
    /// gv1: R.Tensor((m, n), dtype="float32") = R.add(x, gv0)

    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(_blocks[0], tvm::PrinterConfig{}));
    /// Output:
    /// m = T.int64()
    /// n = T.int64()
    /// x: R.Tensor((m, n), dtype="float32")
    /// gv0: R.Tensor((m, n), dtype="float32")
    /// gv1: R.Tensor((m, n), dtype="float32") = R.add(x, gv0)

    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(seq_expr, tvm::PrinterConfig{}));
    /// Output:
    /// m = T.int64()
    /// n = T.int64()
    /// x: R.Tensor((m, n), dtype="float32")
    /// gv0: R.Tensor((m, n), dtype="float32")
    /// gv1: R.Tensor((m, n), dtype="float32") = R.add(x, gv0)
    /// gv1
  }

  /// If & WithFields
  ///
  /// @code{.python}
  /// import tvm
  /// from tvm import relax as rx
  /// from tvm import tir
  /// from tvm.script import ir as I, relax as R, tir as T
  ///
  /// m = tir.Var("m", "int64")
  /// n = tir.Var("n", "int64")
  /// x = rx.Var("x", R.Tensor([m, n], "float32"))
  /// cond = rx.Var("cond", R.Tensor([], "bool"))
  ///
  /// # v_in_if is invisible in the outer scope
  /// v_in_if = rx.Var("v_in_if", R.Tensor([m, n], "float32"))
  /// # gv0 is visible in the outer scope
  /// gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
  ///
  /// # build true branch
  /// true_bindings = [
  ///     rx.VarBinding(v_in_if, rx.op.add(x, x)),
  ///     rx.VarBinding(gv0, rx.op.multiply(x, x)),
  /// ]
  /// true_blocks = [rx.BindingBlock(true_bindings)]
  /// true_seq_expr = rx.SeqExpr(true_blocks, true_blocks[-1].bindings[-1].var)
  ///
  /// # build false branch
  /// false_bindings = [
  ///     rx.VarBinding(v_in_if, rx.op.multiply(x, x)),
  ///     rx.VarBinding(gv0, rx.op.add(x, x)),
  /// ]
  /// false_blocks = [rx.BindingBlock(false_bindings)]
  /// false_seq_expr = rx.SeqExpr(false_blocks, false_blocks[-1].bindings[-1].var)
  ///
  /// # build If node
  /// if_node = rx.If(cond=cond, true_branch=true_seq_expr, false_branch=false_seq_expr)
  /// # Output:
  /// # cond: R.Tensor((), dtype="bool")
  /// # m = T.int64()
  /// # n = T.int64()
  /// # if cond:
  /// #     x: R.Tensor((m, n), dtype="float32")
  /// #     v_in_if: R.Tensor((m, n), dtype="float32") = R.add(x, x)
  /// #     gv0: R.Tensor((m, n), dtype="float32") = R.multiply(x, x)
  /// #     gv0
  /// # else:
  /// #     x: R.Tensor((m, n), dtype="float32")
  /// #     v_in_if: R.Tensor((m, n), dtype="float32") = R.multiply(x, x)
  /// #     gv0: R.Tensor((m, n), dtype="float32") = R.add(x, x)
  /// #     gv0
  /// @endcode
  {
    tvm::tir::Var m{"m", DataType::Int(64, 1)};
    tvm::tir::Var n{"n", DataType::Int(64, 1)};
    Var x{
        "x", TensorStructInfo(ShapeExpr{{m, n}},
         DataType::Float(32, 1))
    };
    Var cond{"cond", TensorStructInfo(ShapeExpr{{PrimExpr{-1}}}, DataType::Bool())};

    // NOLINTNEXTLINE
    Var v_in_f{
        "v_in_f", TensorStructInfo{ShapeExpr{{m, n}}, DataType::Float(32, 1)}
    };
    Var gv0{
        "gv0", TensorStructInfo{ShapeExpr{{m, n}}, DataType::Float(32, 1)}
    };

    OpRegistry *opregistry = OpRegistry::Global();
    // NOLINTNEXTLINE
    Call call_add{
        Op::Get("relax.add"), {x, x}
    };
    // NOLINTNEXTLINE
    Call call_multiply{
        Op::Get("relax.multiply"), {x, x}
    };
    // NOLINTNEXTLINE
    tvm::Array<Binding> true_bindings{
        {VarBinding{v_in_f, call_add}, VarBinding{gv0, call_multiply}}
    };
    // NOLINTNEXTLINE
    tvm::Array<BindingBlock> true_blocks{{BindingBlock{true_bindings}}};
    // NOLINTNEXTLINE
    SeqExpr true_seq_expr{true_blocks, true_blocks[0]->bindings[1]->var};
    // NOLINTNEXTLINE
    tvm::Array<Binding> false_bindings{
        {VarBinding{v_in_f, call_multiply}, VarBinding{gv0, call_add}}
    };
    // NOLINTNEXTLINE
    tvm::Array<BindingBlock> false_blocks{{BindingBlock{false_bindings}}};
    // NOLINTNEXTLINE
    SeqExpr false_seq_expr{false_blocks, false_blocks[0]->bindings[1]->var};
    // NOLINTNEXTLINE
    If if_node{cond, true_seq_expr, false_seq_expr};

    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(if_node, tvm::PrinterConfig{}));
    /// Output:
    /// cond: R.Tensor((-1,), dtype="bool")
    /// m = T.int64()
    /// n = T.int64()
    /// if cond:
    ///     x: R.Tensor((m, n), dtype="float32")
    ///     v_in_f: R.Tensor((m, n), dtype="float32") = R.add(x, x)
    ///     gv0: R.Tensor((m, n), dtype="float32") = R.multiply(x, x)
    ///     gv0
    /// else:
    ///     x: R.Tensor((m, n), dtype="float32")
    ///     v_in_f: R.Tensor((m, n), dtype="float32") = R.multiply(x, x)
    ///     gv0: R.Tensor((m, n), dtype="float32") = R.add(x, x)
    ///     gv0

    If ifnode = WithFields(if_node, cond, true_seq_expr, false_seq_expr);
    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(ifnode, tvm::PrinterConfig{}));
  }

  /// Function
  ///
  /// @code{.python}
  /// import tvm
  /// import tvm.relax as rx
  ///
  /// scalar_struct_info = rx.TensorStructInfo(shape=[], dtype="int32")
  /// gv0 = rx.Var("gv0", scalar_struct_info)
  /// f = rx.Var("f", rx.FuncStructInfo([scalar_struct_info], scalar_struct_info))
  /// ipt = rx.Var("ipt", scalar_struct_info)
  /// x0 = rx.Var("x0", scalar_struct_info)
  /// x1 = rx.Var("x1", scalar_struct_info)
  /// x2 = rx.Var("x2", scalar_struct_info)
  /// y = rx.Var("y", scalar_struct_info)
  /// inner_block = rx.BindingBlock(
  ///     [rx.VarBinding(x0, rx.const(2, "int32")), rx.VarBinding(y, rx.Call(f, [x0]))]
  /// )
  /// inner_func = rx.Function([ipt], rx.SeqExpr([inner_block], y), scalar_struct_info)
  /// outer_block = rx.BindingBlock(
  ///     [
  ///         rx.VarBinding(f, inner_func),
  ///         rx.VarBinding(x1, rx.const(1, "int32")),
  ///         rx.VarBinding(x2, rx.op.add(x1, rx.Call(f, [x1]))),
  ///         rx.VarBinding(gv0, x2),
  ///     ]
  /// )
  /// func = rx.Function([], rx.SeqExpr([outer_block], gv0), scalar_struct_info)
  /// func.show()
  /// # Output:
  /// # # from tvm.script import relax as R
  /// #
  /// # @R.function(private=True)
  /// # def main() -> R.Tensor((), dtype="int32"):
  /// #     # from tvm.script import relax as R
  /// #
  /// #     @R.function
  /// #     def f(ipt: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
  /// #         x0: R.Tensor((), dtype="int32") = R.const(2, "int32")
  /// #         y: R.Tensor((), dtype="int32") = f(x0)
  /// #         return y
  /// #
  /// #     x1: R.Tensor((), dtype="int32") = R.const(1, "int32")
  /// #     x2: R.Tensor((), dtype="int32") = R.add(x1, f(x1))
  /// #     gv0: R.Tensor((), dtype="int32") = x2
  /// #     return gv0
  /// @endcode
  {
    // NOLINTNEXTLINE
    TensorStructInfo scalar_struct_info{ShapeExpr{{PrimExpr{-1}}}, DataType::Int(32, 1)};
    Var gv0{"gv0", scalar_struct_info};
    Var f{
        "f", FuncStructInfo{{scalar_struct_info}, scalar_struct_info}
    };
    Var ipt{"ipt", scalar_struct_info};
    Var x0{"x0", scalar_struct_info};
    Var x1{"x1", scalar_struct_info};
    Var x2{"x2", scalar_struct_info};
    Var y{"y", scalar_struct_info};

    NDArray ndarrayconstvalue2 =
        NDArray::Empty(ShapeTuple{1}, DLDataType{DLDataTypeCode::kDLInt, 32, 1},
                       DLDevice{DLDeviceType::kDLCPU, 0});
    int constvalue2 = 2;
    ndarrayconstvalue2.CopyFromBytes(&constvalue2, sizeof(int));

    NDArray ndarrayconstvalue1 =
        NDArray::Empty(ShapeTuple{1}, DLDataType{DLDataTypeCode::kDLInt, 32, 1},
                       DLDevice{DLDeviceType::kDLCPU, 0});
    int constvalue1 = 1;
    ndarrayconstvalue1.CopyFromBytes(&constvalue1, sizeof(int));

    // LOG_PRINT_VAR(*static_cast<int *>(ndarrayconstvalue2.operator->()->data));
    // LOG_PRINT_VAR(*static_cast<int *>(ndarrayconstvalue1.operator->()->data));
    Constant constant2{ndarrayconstvalue2};
    Constant constant1{ndarrayconstvalue1};
    LOG_PRINT_VAR(GetShapeOf(constant2));
    LOG_PRINT_VAR(GetShapeOf(constant1));
    // NOLINTNEXTLINE
    BindingBlock inner_block{
        {VarBinding{x0, constant2}, VarBinding{y, Call{f, {x0}}}}
    };
    // NOLINTNEXTLINE
    Function inner_func{{ipt}, SeqExpr({inner_block}, y), scalar_struct_info};

    OpRegistry *opregistry = OpRegistry::Global();
    // NOLINTNEXTLINE
    Call call_add{
        Op::Get("relax.add"), {x1, Call{f, {x1}}}
    };
    // NOLINTNEXTLINE
    BindingBlock outer_block{
        {VarBinding{f, inner_func}, VarBinding{x1, constant1}, VarBinding{x2, call_add},
         VarBinding{gv0, x2}}
    };

    Function func{{}, SeqExpr({outer_block}, gv0), scalar_struct_info};

    LOG_PRINT_VAR(tvm::TVMScriptPrinter::Script(func, tvm::PrinterConfig{}));
    /// Output:
    /// # from tvm.script import relax as R
    ///
    /// @R.function(private=True)
    /// def main() -> R.Tensor((-1,), dtype="int32"):
    ///     # from tvm.script import relax as R
    ///
    ///    @R.function
    ///    def f(ipt: R.Tensor((-1,), dtype="int32")) -> R.Tensor((-1,), dtype="int32"):
    ///        x0: R.Tensor((-1,), dtype="int32") = metadata["relax.expr.Constant"][0]
    ///        y: R.Tensor((-1,), dtype="int32") = f(x0)
    ///        return y
    ///
    ///    x1: R.Tensor((-1,), dtype="int32") = metadata["relax.expr.Constant"][1]
    ///    x2: R.Tensor((-1,), dtype="int32") = R.add(x1, f(x1))
    ///    gv0: R.Tensor((-1,), dtype="int32") = x2
    ///    return gv0
  }

  /// Externfunc
  {
    ExternFunc externfunc{"vm.builtin.tensor_to_shape"};
    LOG_PRINT_VAR(externfunc);
  }
}

}  // namespace expr_test

void CallTest() { expr_test::CallTest(); }
void TupleTest() { expr_test::TupleTest(); }
void TupleGetItemTest() { expr_test::TupleGetItemTest(); }
void LeafExprTest() { expr_test::LeafExprTest(); }
void BindTest() { expr_test::BindTest(); }
