#include "tir/var-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>

namespace var_test {

void VarTest() {
  LOG_SPLIT_LINE("VarTest");
  Var x{"var", DataType::UInt(32, 1, false)};
  LOG_PRINT_VAR(x->name_hint);        // "var"
  LOG_PRINT_VAR(x->dtype);            // uint32
  LOG_PRINT_VAR(x->type_annotation);  // T.uint32
  LOG_BLANK_LINE;

  Var y = x.copy_with_name("varcopy_with_name");
  LOG_PRINT_VAR(y->name_hint);        // => "varcopy_with_name"
  LOG_PRINT_VAR(y->dtype);            // default to uint32
  LOG_PRINT_VAR(y->type_annotation);  // default to T.uint32
  LOG_BLANK_LINE;

  Var z = x.copy_with_suffix("suffix");
  LOG_PRINT_VAR(z->name_hint);        // => "varsuffix"
  LOG_PRINT_VAR(z->dtype);            // default to uint32
  LOG_PRINT_VAR(z->type_annotation);  // default to T.uint32
  LOG_BLANK_LINE;

  Var w = x.copy_with_dtype(DataType::Int(16, 1));
  LOG_PRINT_VAR(w->name_hint);        // default to "var"
  LOG_PRINT_VAR(w->dtype);            // => int16
  LOG_PRINT_VAR(w->type_annotation);  // => T.int16
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.get() == y.get());
}

void SizeVarTest() {
  LOG_SPLIT_LINE("SizeVarTest");
  SizeVar x{"sizevar", DataType::UInt(32, 1, false)};
  LOG_PRINT_VAR(x->name_hint);
  LOG_PRINT_VAR(x->dtype);
  LOG_PRINT_VAR(x->type_annotation);
  LOG_BLANK_LINE;

  Var y = x.copy_with_name("varcopy_with_name");
  LOG_PRINT_VAR(y->name_hint);
  LOG_PRINT_VAR(y->dtype);
  LOG_PRINT_VAR(y->type_annotation);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.get() == y.get());
  LOG_BLANK_LINE;

  Var z = x.copy_with_suffix("suffix");
  LOG_PRINT_VAR(z->name_hint);
  LOG_PRINT_VAR(z->dtype);
  LOG_PRINT_VAR(z->type_annotation);
  LOG_BLANK_LINE;

  Var w = x.copy_with_dtype(DataType::Int(32, 1));
  LOG_PRINT_VAR(w->name_hint);
  LOG_PRINT_VAR(w->dtype);
  LOG_PRINT_VAR(w->type_annotation);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.get() == w.get());
  LOG_BLANK_LINE;

  SizeVar o{"sizevar", PointerType{VoidType()}};
  LOG_PRINT_VAR(o->name_hint);
  LOG_PRINT_VAR(o->dtype);            // => handle
  LOG_PRINT_VAR(o->type_annotation);  // => T.handle(None)
  LOG_BLANK_LINE;
}

void IterVarTest() {
  LOG_SPLIT_LINE("IterVarTest");

  PrimExpr x = 4;
  PrimExpr y = 4;

  LOG_PRINT_VAR(x.as<IntImmNode>()->value);
  LOG_PRINT_VAR(y.as<IntImmNode>()->value);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.as<PrimExprNode>()->dtype);
  LOG_PRINT_VAR(y.as<PrimExprNode>()->dtype);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.as<BaseExprNode>()->GetTypeKey());
  LOG_PRINT_VAR(y.as<BaseExprNode>()->GetTypeKey());
  LOG_BLANK_LINE;

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

  LOG_PRINT_VAR(itervar->dom);
  LOG_PRINT_VAR(itervar->dom->extent.defined());
  LOG_PRINT_VAR(itervar->var);
  LOG_PRINT_VAR(itervar->iter_type);
  LOG_PRINT_VAR(IterVarType2String(itervar->iter_type));
  LOG_PRINT_VAR(itervar->thread_tag);
  LOG_PRINT_VAR(itervar->span);

  LOG_PRINT_VAR(itervar.as<IterVarNode>()->dom);
}

}  // namespace var_test

REGISTER_TEST_SUITE(var_test::VarTest, tir_var_test_VarTest);
REGISTER_TEST_SUITE(var_test::SizeVarTest, tir_var_test_SizeVarTest);
REGISTER_TEST_SUITE(var_test::IterVarTest, tir_var_test_IterVarTest);
