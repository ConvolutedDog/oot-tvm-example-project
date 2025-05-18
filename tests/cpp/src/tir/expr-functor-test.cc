#include "tir/expr-functor-test.h"
#include "test-func-registry.h"
#include "tvm/tir/op.h"

namespace expr_functor_test {

void ExprFunctorTest() {
  LOG_SPLIT_LINE("ExprFunctorTest");

  ExprVisitor visitor;

  visitor(tvm::floor(1.2f));
  visitor(tvm::tir::const_true(4));
  visitor(tvm::tir::MakeConstScalar(tvm::DataType::Float(32), 1.0f));

  int n = 10;
  tvm::tir::IterVar i = tvm::tir::IterVar(tvm::Range(0, n), tvm::tir::Var{"i"},
                                          tvm::tir::IterVarType::kCommReduce);
  tvm::tir::IterVar j = tvm::tir::IterVar(tvm::Range(0, n), tvm::tir::Var{"j"},
                                          tvm::tir::IterVarType::kCommReduce);
  tvm::runtime::Array<tvm::tir::IterVar> rdom = {i, j};
  tvm::runtime::Array<tvm::PrimExpr> init = {
      tvm::tir::make_const(tvm::DataType::Int(32), 0)};
  tvm::tir::Buffer buffer = tvm::tir::decl_buffer({n, n}, tvm::DataType::Int(32), "A");
  tvm::PrimExpr source = tvm::tir::BufferLoad(buffer, {i, j});
  // $totalsum = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} A[i][j]$.
  tvm::PrimExpr totalsum = sum(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalsum);
  tvm::PrimExpr totalmin = min(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalmin);
  tvm::PrimExpr totalmax = max(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalmax);
  tvm::PrimExpr totalprod = prod(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalprod);

  visitor(totalprod);

  /// @todo (yangjianchao) Supplement a derived class.
}

}  // namespace expr_functor_test

REGISTER_TEST_SUITE(expr_functor_test::ExprFunctorTest,
                    tir_expr_functor_test_ExprFunctorTest);
