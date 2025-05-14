#include "node/functor-test.h"
#include "test-func-registry.h"
#include <string>
#include <tvm/ir/expr.h>

namespace functor_test {

void NodeFunctorTest() {
  LOG_SPLIT_LINE("NodeFunctorTest");

  NodeFunctor<std::string(const ObjectRef &n, std::string s)> nodefunctor;
  nodefunctor.set_dispatch<tvm::IntImmNode>([](const ObjectRef &n, std::string s) {
    return n->GetTypeKey() + " : IntImmNode -> " + std::move(s);
  });
  nodefunctor.set_dispatch<tvm::FloatImmNode>([](const ObjectRef &n, std::string s) {
    return n->GetTypeKey() + " : FloatImmNode -> " + std::move(s);
  });

  tvm::PrimExpr a = 3;
  tvm::PrimExpr b = 4;
  tvm::PrimExpr c = a + b;

  tvm::FloatImm y{tvm::runtime::DataType::Float(32), 3.0f};

  LOG_PRINT_VAR(nodefunctor(c, "NodeFunctor Int"));
  LOG_PRINT_VAR(nodefunctor(y, "NodeFunctor Float"))

  LOG_PRINT_VAR(nodefunctor.can_dispatch(c));
  LOG_PRINT_VAR(nodefunctor.can_dispatch(y));

  nodefunctor.clear_dispatch<tvm::IntImmNode>();
  nodefunctor.Finalize();

  LOG_PRINT_VAR(nodefunctor.can_dispatch(c));
  LOG_PRINT_VAR(nodefunctor.can_dispatch(y));
}

}  // namespace functor_test

void NodeFunctorTest() { functor_test::NodeFunctorTest(); }

namespace {

REGISTER_TEST_SUITE(NodeFunctorTest);

}
