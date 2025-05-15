#include "ir/type-functor-test.h"
#include "test-func-registry.h"

namespace type_functor_test {

R MyTypeFunctor::VisitType_(const FuncTypeNode *op) { return op->arg_types.size(); }

R MyTypeFunctor::VisitType_(const PointerTypeNode *op) {
  return op->storage_scope.size();
}

R MyTypeFunctor::VisitType_(const PrimTypeNode *op) { return op->dtype.code(); }

R MyTypeFunctor::VisitType_(const TupleTypeNode *op) { return op->fields.size(); }

void TypeFunctorTest() {
  LOG_SPLIT_LINE("TypeFunctorTest");

  MyTypeFunctor functor;
  PrimType primtype{
      DataType{2, 32, 1}
  };
  CHECK_EQ(functor(primtype), 2);
  LOG_PRINT_VAR(functor(primtype));

  TupleType tupletype{
      Array<Type>{primtype, primtype}
  };
  CHECK_EQ(functor(tupletype), 2);
  LOG_PRINT_VAR(functor(tupletype));
}

}  // namespace type_functor_test

REGISTER_TEST_SUITE(type_functor_test::TypeFunctorTest,
                    ir_type_functor_test_TypeFunctorTest);
