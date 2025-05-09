#include "runtime/object-test.h"
#include "tvm/runtime/memory.h"
#include <tvm/runtime/object.h>

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

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

namespace objectref_test {

TestCanDerivedFrom2::TestCanDerivedFrom2(String name) {
  ObjectPtr<TestCanDerivedFromNode> n = make_object<TestCanDerivedFromNode>();
  n->nameHint = std::move(name);
  data_ = std::move(n);
}

/// Although TestDerived3 inherits from TestCanDerivedFrom2, it cannot just
/// directly use the constructor function of TestCanDerivedFrom2, because it
/// has different internal Object type. And to achive the initialization of
/// name which is inheritted from TestCanDerivedFrom2, we should also provide
/// a default constructor function for TestCanDerivedFrom2.
TestDerived3::TestDerived3(String name, String extraName) {
  ObjectPtr<TestDerived3Node> n = make_object<TestDerived3Node>();
  n->nameHint = std::move(name);
  n->extraNameHint = std::move(extraName);
  data_ = std::move(n);
}

}  // namespace objectref_test

std::ostream &operator<<(std::ostream &os, const tvm::runtime::ObjectRef &clsref) {
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
  using objectref_test::TestDerived3;
  using objectref_test::TestFinal;
  using tvm::runtime::make_object;
  using tvm::runtime::ObjectPtr;

  using objectref_test::TestCanDerivedFrom2;

  ObjectPtr<TestCanDerivedFromNode> objptr = make_object<TestCanDerivedFromNode>();
  TestCanDerivedFrom testCanDerivedFromRef(objptr);
  LOG_SPLIT_LINE("testCanDerivedFromRef");
  std::cout << testCanDerivedFromRef << '\n';

  ObjectPtr<TestDerived1Node> objptrchild1 = make_object<TestDerived1Node>();
  TestDerived1 testDerived1Ref(objptrchild1);
  LOG_SPLIT_LINE("testDerived1Ref");
  std::cout << testDerived1Ref << '\n';

  ObjectPtr<TestDerived2Node> objptrchild2 = make_object<TestDerived2Node>();
  TestDerived2 testDerived2Ref(objptrchild2);
  LOG_SPLIT_LINE("testDerived2Ref");
  std::cout << testDerived2Ref << '\n';

  TestFinal testFinalRef(make_object<TestFinalNode>());
  LOG_SPLIT_LINE("testFinalRef");
  std::cout << testFinalRef << '\n';

  /// Different ObjectPtr<TestCanDerivedFromNode>
  TestCanDerivedFrom testCanDerivedFromRef2(
      make_object<TestCanDerivedFromNode>(*(testCanDerivedFromRef.get())));
  LOG_PRINT_VAR(testCanDerivedFromRef2 == testCanDerivedFromRef);  // False

  /// Same ObjectPtr<TestCanDerivedFromNode>
  TestCanDerivedFrom testCanDerivedFromRef3(objptr);
  LOG_PRINT_VAR(testCanDerivedFromRef3 == testCanDerivedFromRef);        // True
  LOG_PRINT_VAR(testCanDerivedFromRef3.same_as(testCanDerivedFromRef));  // True
  LOG_PRINT_VAR(testCanDerivedFromRef3.use_count());                     // 3
  LOG_PRINT_VAR("\n");

  /// as Self
  TestDerived1 testDerived1Ref2(objptrchild1);
  LOG_SPLIT_LINE("testDerived1Ref2");
  std::cout << *(testDerived1Ref2.as<TestDerived1Node>()) << '\n';

  /// We can use `Ref.as<Node>()` to transfer a `ObjectRef` object to a `Node *`
  /// object.
  std::cout << *(testDerived1Ref2.as<TestCanDerivedFromNode>()) << '\n';

  /// We can use GetObjectPtr to get a ObjectPtr<Node> object of a node.
  TestCanDerivedFromNode x = object_test::InitObject<TestCanDerivedFromNode>();
  ObjectPtr<TestCanDerivedFromNode> xptr =
      tvm::runtime::GetObjectPtr<TestCanDerivedFromNode>(&x);
  TestCanDerivedFrom testXRef(xptr);
  LOG_SPLIT_LINE("testXRef");
  std::cout << testXRef << '\n';

  /// Customize the initialization of an ObjectRef object.
  TestCanDerivedFrom2 testCanDerivedFrom2Ref{"varname"};
  LOG_SPLIT_LINE("testCanDerivedFrom2Ref");
  std::cout << testCanDerivedFrom2Ref << '\n';
  LOG_PRINT_VAR(testCanDerivedFrom2Ref.get()->nameHint);

  TestDerived3 testDerived3Ref = TestDerived3("varname3", "extraname3");
  LOG_SPLIT_LINE("testDerived3Ref");
  std::cout << testDerived3Ref << '\n';
  LOG_PRINT_VAR(testDerived3Ref.get()->nameHint);
  LOG_PRINT_VAR(testDerived3Ref.get()->extraNameHint);

  /// @important `Ref.get()->` can be written to `Ref->`. 
  LOG_PRINT_VAR(testDerived3Ref.get()->nameHint);
  LOG_PRINT_VAR(testDerived3Ref->extraNameHint);
}
