#include "tir/expr-functor-test.h"
#include "test-func-registry.h"
#include <tvm/tir/expr_functor.h>

namespace expr_functor_test {

void ExprFunctorTest() {
  LOG_SPLIT_LINE("ExprFunctorTest");

  ExprVisitor visitor;

  visitor(floor(1.2f));
  visitor(const_true(4));
  visitor(MakeConstScalar(DataType::Float(32), 1.0f));

  int n = 10;
  IterVar i = IterVar(Range(0, n), Var{"i"}, IterVarType::kCommReduce);
  IterVar j = IterVar(Range(0, n), Var{"j"}, IterVarType::kCommReduce);
  Array<IterVar> rdom = {i, j};
  Array<PrimExpr> init = {make_const(DataType::Int(32), 0)};
  Buffer buffer = decl_buffer({n, n}, DataType::Int(32), "A");
  PrimExpr source = BufferLoad(buffer, {i, j});
  // $totalsum = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} A[i][j]$.
  PrimExpr totalsum = sum(source, rdom, init, Span());
  LOG_PRINT_VAR(totalsum);
  PrimExpr totalmin = min(source, rdom, init, Span());
  LOG_PRINT_VAR(totalmin);
  PrimExpr totalmax = max(source, rdom, init, Span());
  LOG_PRINT_VAR(totalmax);
  PrimExpr totalprod = prod(source, rdom, init, Span());
  LOG_PRINT_VAR(totalprod);

  visitor(totalprod);

  StdCoutExprVisitor stdcoutvisitor;
  LOG_PRINT_VAR_ONLY(stdcoutvisitor(tvm::tir::Var{"i"}));
}

/// Define VisitExpr methods for class `StdCoutExprVisitor`.
using R = StdCoutExprVisitor::R;

R StdCoutExprVisitor::VisitExpr_(const VarNode *op) {
  return CastString("VarNode:", op->name_hint, op->dtype);
}

R StdCoutExprVisitor::VisitExprDefault_(const Object *op) {
  LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
}

/// Cast Everything to a String.
template <typename... Args> String CastString(Args... args) {
  return (... + (Everything2String(args) + String(" ")));
}

}  // namespace expr_functor_test

REGISTER_TEST_SUITE(expr_functor_test::ExprFunctorTest,
                    tir_expr_functor_test_ExprFunctorTest);
