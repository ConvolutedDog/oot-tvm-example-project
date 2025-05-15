#include "node/functor-test.h"
#include "test-func-registry.h"
#include <string>
#include <tvm/ir/expr.h>

namespace functor_test {

void NodeFunctorTest() {
  LOG_SPLIT_LINE("NodeFunctorTest");

  /// @brief `NodeFunctor` is a template class that can dispatch to different functions
  /// based on the type of the input node. It has serveral template params: the first
  /// param is `R` which means the return type of these functions, and `typename... Args`
  /// which means the types of the remaining arguments. Once specified the return type `R`
  /// and the types of the remaining arguments, the `NodeFunctor` class will store many
  /// functions for different types of nodes. For example, we can define a `NodeFunctor`
  /// that are used to print `ObjectRef` instances:
  ///     NodeFunctor<std::string(const ObjectRef &n, std::string prefix)> functor;
  /// In the internal of `NodeFunctor`, when users call:
  ///     functor.set_dispatch<SumObjectRef>([](const ObjectRef &n, std::string s) {...})
  /// the `functor` will store this lambda function in `NodeFunctor::func_`, and the index
  /// of this function (this function corresponds to `SumObjectRef` and can only be called
  /// when the first input of `functor(SumObjectRef n, std::string s)` is `SumObjectRef`)
  /// will be the `type_index()` of `SumObjectRef`: `SumObjectRef->type_index()`. Please
  /// note that we temporarily don't take `Finalize()` in account, which may cause the
  /// index to be not of the `type_index()`. Then users can use `functor(SumObjectRef n,
  /// std::string s)` to call this function. And user can use `can_dispatch(ObjectRef)`
  /// to check whether the functor can dispatch the given `ObjectRef` instance.
  ///
  /// This is also a reflection mechanism.

  /// Define a dispatch functor that takes `ObjectRef` as the first input and
  /// `std::string` and returns `std::string`.
  NodeFunctor<std::string(const ObjectRef &n, std::string s)> nodefunctor;

  /// Set the function for `IntImmNode`.
  nodefunctor.set_dispatch<tvm::IntImmNode>([](const ObjectRef &n, std::string s) {
    return n->GetTypeKey() + " : IntImmNode -> " + std::move(s);
  });

  /// Set the function for `FloatImmNode`.
  nodefunctor.set_dispatch<tvm::FloatImmNode>([](const ObjectRef &n, std::string s) {
    return n->GetTypeKey() + " : FloatImmNode -> " + std::move(s);
  });

  tvm::PrimExpr a = 3;
  tvm::PrimExpr b = 4;
  tvm::PrimExpr c = a + b;

  tvm::FloatImm y{tvm::runtime::DataType::Float(32), 3.0f};

  /// Call the corresponding function in `nodefunctor`.
  LOG_PRINT_VAR(nodefunctor(c, "NodeFunctor Int"));
  LOG_PRINT_VAR(nodefunctor(y, "NodeFunctor Float"))

  /// Check if the function for `c: tvm::PrimExpr` and `y: tvm::FloatImm` is registered.
  LOG_PRINT_VAR(nodefunctor.can_dispatch(c));
  LOG_PRINT_VAR(nodefunctor.can_dispatch(y));

  /// Clear the dispatch function for `tvm::IntImmNode`.
  nodefunctor.clear_dispatch<tvm::IntImmNode>();

  /// Finalize the functor after calling sequence of `set_dispatch`. This function will
  /// attempt to find the min type index that is not null and optimize the space of the
  /// func table so it is more compact.
  ///
  /// For easily, this function will delete the slots that are from 0 to the min index
  /// of the non-null slots in `NodeFunctor::func_`, and maintains a variable called
  /// `begin_type_index_` to save the min index. This can save space to some extent.
  /// But it will not save all the space for the reason that it uses `type_index()` as
  /// the index of the `NodeFunctor::func_`.
  nodefunctor.Finalize();

  LOG_PRINT_VAR(nodefunctor.can_dispatch(c));
  LOG_PRINT_VAR(nodefunctor.can_dispatch(y));
}

}  // namespace functor_test

void NodeFunctorTest() { functor_test::NodeFunctorTest(); }

namespace {

REGISTER_TEST_SUITE(NodeFunctorTest);

}
