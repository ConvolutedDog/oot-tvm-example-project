#include "tvm/runtime/object.h"

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

template <typename T> inline T InitObject() {
  static_assert(std::is_base_of<tvm::runtime::Object, T>::value,
                "can only be used to init Object");
  return T::Create();
}

template <typename Derived> class CreateHelper {
public:
  static Derived Create() {
    Derived obj{};
    obj.type_index_ = Derived::RuntimeTypeIndex();
    return obj;
  }
};

class TestCanDerivedFromObject : public tvm::runtime::Object,
                                 public CreateHelper<TestCanDerivedFromObject> {
public:
  friend class CreateHelper<TestCanDerivedFromObject>;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestCanDerivedFromObject";
  static const constexpr int _type_child_slots = 2;
  static const constexpr bool _type_child_slots_can_overflow = 0;
  TVM_DECLARE_BASE_OBJECT_INFO(TestCanDerivedFromObject, tvm::runtime::Object);
};

class TestDerived : public TestCanDerivedFromObject,
                    public CreateHelper<TestDerived> {
public:
  friend class CreateHelper<TestDerived>;
  /// @note `TestDerived` inherits from `TestCanDerivedFromObject`, which may
  /// cause ambiguity between the `Create` function in
  /// `TestCanDerivedFromObject` and the `Create` function in
  /// `CreateHelper<TestDerived>`. So, we need to explicitly specify which
  /// `Create` function to use.
  using CreateHelper<TestDerived>::Create;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  /// @note `TypeContext::Global()` will return the address of a static `TypeContext`
  /// object, the `std::vector<TypeInfo> type_table_` in `TypeContext` will store
  /// all of the types allocated for each node inheritted from `Object`.
  /// @ref `tvm::runtime::TypeContext::GetOrAllocRuntimeTypeIndex(...)`
  /// @warning `_type_key` is very important because it will be used to find or
  /// allocate runtime `type_index_`. If two class inheritted from `Object` or subclasses
  /// inheritted from `Object` have the same `_type_key`, their `_type_key` will also
  /// be same.
  static constexpr const char *_type_key = "test.TestDerived";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived, TestCanDerivedFromObject);
};

class TestDerived1 : public TestCanDerivedFromObject,
                     public CreateHelper<TestDerived1> {
public:
  friend class CreateHelper<TestDerived1>;
  using CreateHelper<TestDerived1>::Create;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestDerived1";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestDerived1, TestCanDerivedFromObject);
};

class TestFinalObject : public tvm::runtime::Object,
                        public CreateHelper<TestFinalObject> {
public:
  friend class CreateHelper<TestFinalObject>;
  static constexpr const uint32_t _type_index =
      tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestFinalObject";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestFinalObject, tvm::runtime::Object);
};

}  // namespace object_test
