#include "tir/var-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>

namespace var_test {

void VarTest() {
  LOG_SPLIT_LINE("VarTest");
  Var x{"var", DataType::UInt(32, 1, false)};
  LOG_PRINT_VAR(x.operator->()->name_hint);
  LOG_PRINT_VAR(x.operator->()->dtype);
  LOG_PRINT_VAR(x.operator->()->type_annotation);

  Var y = x.copy_with_name("varcopy_with_name");
  LOG_PRINT_VAR(y.operator->()->name_hint);
  LOG_PRINT_VAR(y.operator->()->dtype);
  LOG_PRINT_VAR(y.operator->()->type_annotation);

  Var z = x.copy_with_suffix("suffix");
  LOG_PRINT_VAR(z.operator->()->name_hint);
  LOG_PRINT_VAR(z.operator->()->dtype);
  LOG_PRINT_VAR(z.operator->()->type_annotation);

  Var w = x.copy_with_dtype(DataType::Int(32, 1));
  LOG_PRINT_VAR(w.operator->()->name_hint);
  LOG_PRINT_VAR(w.operator->()->dtype);
  LOG_PRINT_VAR(w.operator->()->type_annotation);

  LOG_PRINT_VAR(x.get() == y.get());
}

void SizeVarTest() {
  LOG_SPLIT_LINE("SizeVarTest");
  SizeVar x{"sizevar", DataType::UInt(32, 1, false)};
  LOG_PRINT_VAR(x.operator->()->name_hint);
  LOG_PRINT_VAR(x.operator->()->dtype);
  LOG_PRINT_VAR(x.operator->()->type_annotation);

  Var y = x.copy_with_name("varcopy_with_name");
  LOG_PRINT_VAR(y.operator->()->name_hint);
  LOG_PRINT_VAR(y.operator->()->dtype);
  LOG_PRINT_VAR(y.operator->()->type_annotation);

  LOG_PRINT_VAR(x.get() == y.get());

  Var z = x.copy_with_suffix("suffix");
  LOG_PRINT_VAR(z.operator->()->name_hint);
  LOG_PRINT_VAR(z.operator->()->dtype);
  LOG_PRINT_VAR(z.operator->()->type_annotation);

  Var w = x.copy_with_dtype(DataType::Int(32, 1));
  LOG_PRINT_VAR(w.operator->()->name_hint);
  LOG_PRINT_VAR(w.operator->()->dtype);
  LOG_PRINT_VAR(w.operator->()->type_annotation);

  LOG_PRINT_VAR(x.get() == y.get());

  SizeVar o{"sizevar", PointerType{VoidType()}};
  LOG_PRINT_VAR(o.operator->()->name_hint);
  LOG_PRINT_VAR(o.operator->()->dtype);
  LOG_PRINT_VAR(o.operator->()->type_annotation);
}

void IterVarTest() {
  LOG_SPLIT_LINE("IterVarTest");

  PrimExpr x = 4;
  PrimExpr y = 4;

  LOG_PRINT_VAR(x.as<IntImmNode>()->value);
  LOG_PRINT_VAR(y.as<IntImmNode>()->value);

  LOG_PRINT_VAR(x.as<PrimExprNode>()->dtype);
  LOG_PRINT_VAR(y.as<PrimExprNode>()->dtype);

  LOG_PRINT_VAR(x.as<BaseExprNode>()->GetTypeKey());
  LOG_PRINT_VAR(y.as<BaseExprNode>()->GetTypeKey());

  Range range{x, y};
  /// Here, the dtype of range is Int (because PrimExpr x = 4 and y = 4,
  /// 4 is an integer). So the dtype of Var should also be defined as Int,
  /// otherwise, the initialization of IterVar will fail. Another point is
  /// that the dtype should always be Int. @ref
  /// https://github.com/apache/tvm/blob/4ef582a3319f30fac2716091f835e493ec161ffd/src/tir/ir/expr.cc#L170
  /// https://github.com/apache/tvm/blob/4ef582a3319f30fac2716091f835e493ec161ffd/src/tir/ir/expr.cc#L174
  DataType dtype = DataType::Int(32, 1);
  Var var{"var", dtype};
  IterVar itervar{range, var, IterVarType::kOrdered, String("thread_tag")};

  LOG_PRINT_VAR(itervar.get()->dom);
  LOG_PRINT_VAR(itervar.get()->dom->extent.defined());
  LOG_PRINT_VAR(itervar.get()->var);
  LOG_PRINT_VAR(itervar.get()->iter_type);
  LOG_PRINT_VAR(IterVarType2String(itervar.get()->iter_type));
  LOG_PRINT_VAR(itervar.get()->thread_tag);
  LOG_PRINT_VAR(itervar.get()->span);

  LOG_PRINT_VAR(itervar.as<IterVarNode>()->dom);
}

}  // namespace var_test

REGISTER_TEST_SUITE(var_test::VarTest, tir_var_test_VarTest);
REGISTER_TEST_SUITE(var_test::SizeVarTest, tir_var_test_SizeVarTest);
REGISTER_TEST_SUITE(var_test::IterVarTest, tir_var_test_IterVarTest);
