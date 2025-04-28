#include "tvm/runtime/container/string.h"
#include "tvm/runtime/memory.h"
#include "tvm/runtime/object.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                   \
  std::cout << "==============" << (stmt) << "==============\n";

/* clang-format off */
/* Sub-class of objects should declare the following static constexpr fields:
 *
 * - _type_index:
 *      Static type index of the object, if assigned to TypeIndex::kDynamic
 *      the type index will be assigned during runtime.
 *      Runtime type index can be accessed by ObjectType::TypeIndex();
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

template <typename T> inline T InitObject() {
  static_assert(std::is_base_of<tvm::runtime::Object, T>::value,
                "can only be used to init Object");
  return T::Create();
}

template <typename Derived> class CreateHelper {
public:
  static Derived Create() {
    Derived obj{};
    /// Once we call this, it will call the `_GetOrAllocRuntimeTypeIndex()`
    /// function and read a static variable `tindex` which is defined in the
    /// scope of `_GetOrAllocRuntimeTypeIndex()`. `tindex` stores the runtime
    /// `type_index_` allocated during the initialization of node.
    obj.type_index_ = Derived::RuntimeTypeIndex();
    return obj;
  }
};

class TestCanDerivedFromNode : public tvm::runtime::Object,
                               public CreateHelper<TestCanDerivedFromNode> {
public:
  String nameHint;

public:
  friend class CreateHelper<TestCanDerivedFromNode>;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestCanDerivedFromNode";
  /// For example, in this header, we have three classes that inherits from
  /// the current class, so there is at least 3 child slots here.
  /// @note The implementation of TVM has bug.
  /// @ref https://github.com/apache/tvm/issues/17901
  static const constexpr int _type_child_slots = 3;
  static const constexpr bool _type_child_slots_can_overflow = 0;
  TVM_DECLARE_BASE_OBJECT_INFO(TestCanDerivedFromNode, tvm::runtime::Object);
};

class TestDerived1Node : public TestCanDerivedFromNode,
                         public CreateHelper<TestDerived1Node> {
public:
  friend class CreateHelper<TestDerived1Node>;
  using CreateHelper<TestDerived1Node>::Create;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestDerived1Node";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived1Node, TestCanDerivedFromNode);
};

class TestDerived2Node : public TestCanDerivedFromNode,
                         public CreateHelper<TestDerived2Node> {
public:
  friend class CreateHelper<TestDerived2Node>;
  /// @note `TestDerived2Node` inherits from `TestCanDerivedFromNode`, which may
  /// cause ambiguity between the `Create` function in
  /// `TestCanDerivedFromNode` and the `Create` function in
  /// `CreateHelper<TestDerived2Node>`. So, we need to explicitly specify which
  /// `Create` function to use.
  using CreateHelper<TestDerived2Node>::Create;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
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

class TestDerived3Node : public TestCanDerivedFromNode,
                         public CreateHelper<TestDerived3Node> {
public:
  String extraNameHint;

public:
  friend class CreateHelper<TestDerived3Node>;
  using CreateHelper<TestDerived3Node>::Create;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestDerived3Node";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived3Node, TestCanDerivedFromNode);
};

class TestFinalNode : public tvm::runtime::Object,
                      public CreateHelper<TestFinalNode> {
public:
  friend class CreateHelper<TestFinalNode>;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestFinalNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestFinalNode, tvm::runtime::Object);
};

}  // namespace object_test

std::ostream &operator<<(std::ostream &os, const tvm::runtime::Object &cls);

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
  TVM_DEFINE_OBJECT_REF_METHODS(TestCanDerivedFrom, ObjectRef,
                                TestCanDerivedFromNode);
};

class TestCanDerivedFrom2 : public ObjectRef {
public:
  TestCanDerivedFrom2() = default;
  explicit TestCanDerivedFrom2(String name);
  const TestCanDerivedFromNode *operator->() const { return get(); }
  const TestCanDerivedFromNode *get() const {
    return static_cast<const TestCanDerivedFromNode *>(data_.get());
  }
  using ContainerType = TestCanDerivedFromNode;
};

class TestDerived1 : public TestCanDerivedFrom {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TestDerived1, TestCanDerivedFrom,
                                TestDerived1Node);
};

class TestDerived2 : public TestCanDerivedFrom {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(TestDerived2, TestCanDerivedFrom,
                                TestDerived2Node);
};

class TestDerived3 : public TestCanDerivedFrom2 {
public:
  explicit TestDerived3(String name, String extraName);
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

}  // namespace objectref_test

std::ostream &operator<<(std::ostream &os,
                         const tvm::runtime::ObjectRef &clsref);

void ObjectTest();
void ObjectRefTest();
