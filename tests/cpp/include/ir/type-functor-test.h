#include "tvm/ir/type.h"
#include "tvm/ir/type_functor.h"
#include <tvm/ir/expr.h>

namespace type_functor_test {

using tvm::TypeFunctor;
using tvm::TypeMutator;
using tvm::TypeVisitor;

using tvm::FuncTypeNode;
using tvm::PointerTypeNode;
using tvm::PrimType;
using tvm::PrimTypeNode;
using tvm::TupleType;
using tvm::TupleTypeNode;
using tvm::Type;

using tvm::runtime::Array;
using tvm::runtime::DataType;
using tvm::runtime::Object;

using R = TypeFunctor<int(const tvm::Type &)>::result_type;

class MyTypeFunctor : public TypeFunctor<int(const tvm::Type &)> {
public:
  ~MyTypeFunctor() = default;
  R VisitType_(const FuncTypeNode *op) final;
  R VisitType_(const TupleTypeNode *op) final;
  R VisitType_(const PrimTypeNode *op) final;
  R VisitType_(const PointerTypeNode *op) final;
  R VisitTypeDefault_(const Object *op) final {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }
};

void TypeFunctorTest();

}  // namespace type_functor_test

void TypeFunctorTest();
