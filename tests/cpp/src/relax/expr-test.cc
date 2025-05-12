#include "relax/expr-test.h"
#include "dlpack/dlpack.h"
#include "tvm/relax/struct_info.h"
#include <tvm/node/script_printer.h>

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
  NDArray ndarray =
      NDArray::Empty(ShapeTuple{1, 2, 3, 4}, DLDataType{DLDataTypeCode::kDLFloat, 32, 1},
                     DLDevice{DLDeviceType::kDLCPU, 0});
  LOG_PRINT_VAR(ndarray);
  Constant constant{ndarray};
  LOG_PRINT_VAR(constant);

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
    NDArray ndarray =
        NDArray::Empty(ShapeTuple{16, 8}, DLDataType{DLDataTypeCode::kDLInt, 64, 1},
                       DLDevice{DLDeviceType::kDLCPU, 0});
    Constant shape{ndarray};
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
}

}  // namespace expr_test

void CallTest() { expr_test::CallTest(); }
void TupleTest() { expr_test::TupleTest(); }
void TupleGetItemTest() { expr_test::TupleGetItemTest(); }
void LeafExprTest() { expr_test::LeafExprTest(); }
void BindTest() { expr_test::BindTest(); }
