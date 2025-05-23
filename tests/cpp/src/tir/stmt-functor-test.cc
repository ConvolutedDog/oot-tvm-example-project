#include "tir/stmt-functor-test.h"
#include "test-func-registry.h"
#include <tvm/tir/stmt_functor.h>

namespace stmt_functor_test {

class StdCoutStmtFunctor : public StmtFunctor<String(const Stmt &n)> {
public:
  using R = StmtFunctor::result_type;
  using StmtFunctor::operator();

  R VisitExpr(const PrimExpr &expr) {
    return "StdCout a PrimExpr: " +
           tvm::runtime::DLDataType2String(expr.dtype().operator DLDataType()) + "\n";
  }

  R VisitStmt_(const LetStmtNode *op) override {
    String ret = "op->var: " + this->VisitExpr(op->var);
    ret = ret + "op->value: " + this->VisitExpr(op->value);
    return ret + "StdCout a LetStmt: " + op->var->name_hint;
  }
};

void TirStmtFunctorTest() {
  LOG_SPLIT_LINE("TirStmtFunctorTest");

  Var x{"x"};
  LetStmt letstmt{x, 1, Evaluate{x}};
  LOG_PRINT_VAR(letstmt);

  StdCoutStmtFunctor visitor;
  LOG_PRINT_VAR(visitor(letstmt));
}

void TirOtherVisitorMutatorTest() {
  LOG_SPLIT_LINE("TirStmtVisitorTest");

  Var x{"x"};
  Var y{"y"};
  LetStmt letstmt{x, 1, Evaluate{x}};
  LOG_PRINT_VAR(letstmt);

  StmtVisitor visitor;
  visitor(letstmt);

  StmtMutator mutator;
  Stmt newstmt = mutator(letstmt);
  LOG_PRINT_VAR(newstmt);

  StmtExprMutator exprmutator;
  LOG_PRINT_VAR(exprmutator(x + y));
}

/// @todo (yangjianchao) Lots of functions.

}  // namespace stmt_functor_test

REGISTER_TEST_SUITE(stmt_functor_test::TirStmtFunctorTest,
                    tir_stmt_functor_test_TirStmtFunctorTest);
REGISTER_TEST_SUITE(stmt_functor_test::TirOtherVisitorMutatorTest,
                    tir_stmt_functor_test_TirOtherVisitorMutatorTest);
