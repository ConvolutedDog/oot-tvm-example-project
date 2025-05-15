#include "tvm/runtime/container/string.h"
#include "tvm/runtime/memory.h"
#include "tvm/runtime/object.h"

/* clang-format off */
/* Sub-class of objects should declare the following static constexpr fields:
 *
 * - _type_index:
 *      Static type index of the object, if assigned to TypeIndex::kDynamic
 *      the type index(`type_index_`) will be assigned during runtime.
 *      Runtime type index(`type_index_`) can be accessed by ObjectType::TypeIndex();
 * - _type_key:
 *       The unique string identifier of the type.
 * - _type_final:
 *       Whether the type is terminal type(there is no subclass of the type in
 *       the object system). This field is automatically set by macro
 *       TVM_DECLARE_FINAL_OBJECT_INFO It is still OK to sub-class a terminal object
 *       type T and construct it using make_object. But IsInstance check will only
 *       show that the object type is T(instead of the sub-class).
 *
 * The following two fields are necessary for base classes that can be
 * sub-classed.
 *
 * - _type_child_slots:
 *       Number of reserved type index slots for child classes.
 *       Used for runtime optimization for type checking in IsInstance.
 *       If an object's type_index is within range of [type_index, type_index +
 *       _type_child_slots] Then the object can be quickly decided as sub-class of the
 *       current object class. If not, a fallback mechanism is used to check the
 *       global type table. Recommendation: set to estimate number of children needed.
 * - _type_child_slots_can_overflow:
 *       Whether we can add additional child classes even if the number of child
 *       classes exceeds the _type_child_slots. A fallback mechanism to check global
 *       type table will be used. Recommendation: set to false for optimal runtime
 *       speed if we know exact number of children.
 *
 * Two macros are used to declare helper functions in the object:
 * - Use TVM_DECLARE_BASE_OBJECT_INFO for object classes that can be sub-classed.
 * - Use TVM_DECLARE_FINAL_OBJECT_INFO for object classes that cannot be sub-classed.
 *
 */
/* clang-format on */

namespace object_test {

using tvm::runtime::String;

class TestCanDerivedFromNode : public tvm::runtime::Object {
private:
  String nameHint;

public:
  // default constructor
  TestCanDerivedFromNode() { type_index_ = RuntimeTypeIndex(); }

  TestCanDerivedFromNode(const String &name) : nameHint(name) {
    type_index_ = RuntimeTypeIndex();
  }
  /**
   * @brief if _type_index is not kDynamic, then the type_index_ = type_index
   * else type_index_ = GetOrAllocRuntimeTypeIndex(type_key)
   */
  static constexpr const uint32_t _type_index = tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestCanDerivedFromNode";
  /// For example, in this header, we have three classes that inherits from
  /// the current class, so there is at least 3 child slots here.
  /// @note The implementation of TVM has bug.
  /// @ref https://github.com/apache/tvm/issues/17901
  static const constexpr int _type_child_slots = 3;
  static const constexpr bool _type_child_slots_can_overflow = 0;
  TVM_DECLARE_BASE_OBJECT_INFO(TestCanDerivedFromNode, tvm::runtime::Object);

  String GetNameHint() const { return nameHint; }
};

class TestDerived1Node : public TestCanDerivedFromNode {
public:
  TestDerived1Node() { type_index_ = RuntimeTypeIndex(); }
  static constexpr const uint32_t _type_index = tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestDerived1Node";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived1Node, TestCanDerivedFromNode);
};

class TestDerived2Node : public TestCanDerivedFromNode {
public:
  TestDerived2Node() { type_index_ = RuntimeTypeIndex(); }
  /// @note `TestDerived2Node` inherits from `TestCanDerivedFromNode`, which may
  /// cause ambiguity between the `Create` function in
  /// `TestCanDerivedFromNode` and the `Create` function in
  /// `CreateHelper<TestDerived2Node>`. So, we need to explicitly specify which
  /// `Create` function to use.
  static constexpr const uint32_t _type_index = tvm::runtime::TypeIndex::kDynamic;
  /// @note `TypeContext::Global()` will return the address of a static
  /// `TypeContext` object, the `std::vector<TypeInfo> type_table_` in
  /// `TypeContext` will store all of the types allocated for each node
  /// inheritted from `Object`.
  /// @ref `tvm::runtime::TypeContext::GetOrAllocRuntimeTypeIndex(...)`
  /// @warning `_type_key` is very important because it will be used to find or
  /// allocate runtime `type_index_`. If two class inheritted from `Object` or
  /// subclasses inheritted from `Object` have the same `_type_key`, their
  /// runtime `_type_index` will also be same.
  static constexpr const char *_type_key = "test.TestDerived2Node";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived2Node, TestCanDerivedFromNode);
};

class TestDerived3Node : public TestCanDerivedFromNode {
private:
  String extraNameHint;

public:
  // constructor
  TestDerived3Node(const String &name, const String &extraName)
      : TestCanDerivedFromNode(name), extraNameHint(extraName) {}
  static constexpr const uint32_t _type_index = tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestDerived3Node";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived3Node, TestCanDerivedFromNode);

  String GetExtraNameHint() const { return extraNameHint; }
};

class TestFinalNode : public tvm::runtime::Object {
public:
  TestFinalNode() { type_index_ = RuntimeTypeIndex(); }
  static constexpr const uint32_t _type_index = tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestFinalNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestFinalNode, tvm::runtime::Object);
};

void ObjectTest();

}  // namespace object_test

namespace objectref_test {

using ::object_test::TestCanDerivedFromNode;
using ::object_test::TestDerived1Node;
using ::object_test::TestDerived2Node;
using ::object_test::TestDerived3Node;
using ::object_test::TestFinalNode;
using ::tvm::runtime::Object;
using ::tvm::runtime::ObjectPtr;
using ::tvm::runtime::ObjectRef;

using tvm::runtime::make_object;
using tvm::runtime::String;

class TestCanDerivedFrom : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TestCanDerivedFrom, ObjectRef, TestCanDerivedFromNode);
};

class TestCanDerivedFrom2 : public ObjectRef {
public:
  TestCanDerivedFrom2() = default;
  explicit TestCanDerivedFrom2(const String &name);
  const TestCanDerivedFromNode *operator->() const { return get(); }
  const TestCanDerivedFromNode *get() const {
    return static_cast<const TestCanDerivedFromNode *>(data_.get());
  }
  using ContainerType = TestCanDerivedFromNode;
};

class TestDerived1 : public TestCanDerivedFrom {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TestDerived1, TestCanDerivedFrom, TestDerived1Node);
};

class TestDerived2 : public TestCanDerivedFrom {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TestDerived2, TestCanDerivedFrom, TestDerived2Node);
};

class TestDerived3 : public TestCanDerivedFrom2 {
public:
  explicit TestDerived3(const String &name, const String &extraName);
  const TestDerived3Node *operator->() const { return get(); }
  const TestDerived3Node *get() const {
    return static_cast<const TestDerived3Node *>(data_.get());
  }
  using ContainerType = TestDerived3Node;
};

class TestFinal : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TestFinal, ObjectRef, TestFinalNode);
};

void ObjectRefTest();

}  // namespace objectref_test
