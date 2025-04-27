#include "../include/object-test.h"
#include "tvm/runtime/memory.h"

namespace object_test {

/// @brief This macros actually calls `::_GetOrAllocRuntimeTypeIndex()`
/// function, this function will calculate a static variable `tindex`.
/// This `tindex` will be returned during the call of function
/// `RuntimeTypeIndex()` which will allocate the runtime `type_index_`
/// for the node that inherits from `Object`. This macro will define a
/// global variable but this variable will never be used, its function
/// is to call this macro to initialize the static `tindex` variable.
/// @note This is not necessary, because it will be called during the
/// initialization of each node if you didn't call this macro here.
TVM_REGISTER_OBJECT_TYPE(TestCanDerivedFromNode);
TVM_REGISTER_OBJECT_TYPE(TestDerived1Node);
TVM_REGISTER_OBJECT_TYPE(TestDerived2Node);
TVM_REGISTER_OBJECT_TYPE(TestFinalNode);

}  // namespace object_test

std::ostream &operator<<(std::ostream &os, const tvm::runtime::Object &cls) {
  LOG_PRINT_VAR(cls.type_index());
  LOG_PRINT_VAR(cls.RuntimeTypeIndex());
  LOG_PRINT_VAR(cls.GetTypeKey());
  LOG_PRINT_VAR(cls.type_index());
  LOG_PRINT_VAR(cls.IsInstance<tvm::runtime::Object>());
  LOG_PRINT_VAR(cls.IsInstance<object_test::TestCanDerivedFromNode>());
  LOG_PRINT_VAR(cls.IsInstance<object_test::TestDerived1Node>());
  LOG_PRINT_VAR(cls.IsInstance<object_test::TestDerived2Node>());
  LOG_PRINT_VAR(cls.IsInstance<object_test::TestFinalNode>());
  LOG_PRINT_VAR(cls.unique());
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const tvm::runtime::ObjectRef &clsref) {
  operator<<(std::cout, *(clsref.get()));
  return os;
}

void ObjectTest() {
  object_test::TestCanDerivedFromNode testCanDerivedFromObj =
      object_test::InitObject<object_test::TestCanDerivedFromNode>();
  LOG_SPLIT_LINE("testCanDerivedFromObj");
  std::cout << testCanDerivedFromObj << '\n';

  object_test::TestDerived1Node testDerived1 =
      object_test::InitObject<object_test::TestDerived1Node>();
  LOG_SPLIT_LINE("testDerived1");
  std::cout << testDerived1 << '\n';

  object_test::TestDerived2Node testDerived2 =
      object_test::InitObject<object_test::TestDerived2Node>();
  LOG_SPLIT_LINE("testDerived2");
  std::cout << testDerived2 << '\n';

  object_test::TestFinalNode testFinalObj =
      object_test::InitObject<object_test::TestFinalNode>();
  LOG_SPLIT_LINE("testFinalObj");
  std::cout << testFinalObj << '\n';
}

void ObjectRefTest() {
  using object_test::InitObject;
  using object_test::TestCanDerivedFromNode;
  using object_test::TestDerived1Node;
  using object_test::TestDerived2Node;
  using object_test::TestFinalNode;
  using objectref_test::TestCanDerivedFrom;
  using objectref_test::TestDerived1;
  using objectref_test::TestDerived2;
  using objectref_test::TestFinal;
  using tvm::runtime::make_object;

  TestCanDerivedFrom testCanDerivedFromRef(make_object<TestCanDerivedFromNode>());
  LOG_SPLIT_LINE("testCanDerivedFromRef");
  std::cout << testCanDerivedFromRef << '\n';

  TestDerived1 testDerived1Ref(make_object<TestDerived1Node>());
  LOG_SPLIT_LINE("testDerived1Ref");
  std::cout << testDerived1Ref << '\n';

  TestDerived2 testDerived2Ref(make_object<TestDerived2Node>());
  LOG_SPLIT_LINE("testDerived2Ref");
  std::cout << testDerived2Ref << '\n';

  TestFinal testFinalRef(make_object<TestFinalNode>());
  LOG_SPLIT_LINE("testFinalRef");
  std::cout << testFinalRef << '\n';
}
