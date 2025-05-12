#include "relax/expr-test.h"
#include "dlpack/dlpack.h"
#include "tvm/relax/struct_info.h"

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

  /// Bind
}

}  // namespace expr_test

void CallTest() { expr_test::CallTest(); }
void TupleTest() { expr_test::TupleTest(); }
void TupleGetItemTest() { expr_test::TupleGetItemTest(); }
void LeafExprTest() { expr_test::LeafExprTest(); }
void BindTest() { expr_test::BindTest(); }
