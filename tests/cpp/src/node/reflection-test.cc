#include "node/reflection-test.h"
#include "test-func-registry.h"
#include "tvm/ir/expr.h"

namespace reflection_test {

std::ostream &operator<<(std::ostream &os, const tvm::runtime::NDArray &arr) {  // NOLINT
  os << "Not implemented.\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, std::vector<std::string> &vec) {
  for (auto &s : vec)
    os << s << " ";
  return os;
}

void AttrVisitorTest() {
  LOG_SPLIT_LINE("AttrVisitorTest");
  PrimExpr start = 4;
  PrimExpr end = 8;

  Range range{start, end};
  Var var{"x", DataType::Int(32)};
  IterVar itervar{range, var, IterVarType::kThreadIndex, "threadidx.x"};

  MyIRSerializer serializer;
  const_cast<IterVarNode *>(itervar.as<IterVarNode>())->VisitAttrs(&serializer);
}

void ReflectionVTableTest() {
  LOG_SPLIT_LINE("ReflectionVTableTest");

  /// @brief `ReflectionVTable` is used to register the vtable of a class. It stores a set
  /// of `FVisitAttrs`/`FSEqualReduce`/`FSHashReduce`/`FCreate`/`FReprBytes` function
  /// pointers and these functions are called using `Object::GetOrAllocRuntimeTypeIndex`:
  ///     /*! \brief Attribute visitor. */
  ///     std::vector<FVisitAttrs> fvisit_attrs_;
  ///     /*! \brief Structural equal function. */
  ///     std::vector<FSEqualReduce> fsequal_reduce_;
  ///     /*! \brief Structural hash function. */
  ///     std::vector<FSHashReduce> fshash_reduce_;
  ///     /*! \brief Creation function. */
  ///     std::vector<FCreate> fcreate_;
  ///     /*! \brief ReprBytes function. */
  ///     std::vector<FReprBytes> frepr_bytes_;
  ///
  /// The first three vectors are registered by `ReflectionVTable::Register()` function,
  /// and the current TVM uses the macro `TVM_REGISTER_REFLECTION_VTABLE` to call the
  /// `ReflectionVTable::Register()` function to achive this goal. For example:
  ///     struct ArrayNodeTrait {
  ///       static constexpr const std::nullptr_t VisitAttrs = nullptr;
  ///       static void SHashReduce(const ArrayNode* key, SHashReducer hash_reduce) {
  ///         hash_reduce(static_cast<uint64_t>(key->size()));
  ///         for (uint32_t i = 0; i < key->size(); ++i) {
  ///           hash_reduce(key->at(i));
  ///         }
  ///       }
  ///       static bool SEqualReduce(const ArrayNode* lhs, const ArrayNode* rhs,
  ///       SEqualReducer equal) {
  ///         if (equal.IsPathTracingEnabled()) {
  ///           return SEqualReduceTraced(lhs, rhs, equal);
  ///         }
  ///         if (lhs->size() != rhs->size()) return false;
  ///         for (uint32_t i = 0; i < lhs->size(); ++i) {
  ///           if (!equal(lhs->at(i), rhs->at(i))) return false;
  ///         }
  ///         return true;
  ///       }
  ///      private:
  ///       static bool SEqualReduceTraced(const ArrayNode* lhs, const ArrayNode* rhs,
  ///                                      const SEqualReducer& equal) {
  ///         uint32_t min_size = std::min(lhs->size(), rhs->size());
  ///         const ObjectPathPair& array_paths = equal.GetCurrentObjectPaths();
  ///         for (uint32_t index = 0; index < min_size; ++index) {
  ///           ObjectPathPair element_paths = {array_paths->lhs_path->ArrayIndex(index),
  ///                                           array_paths->rhs_path->ArrayIndex(index)};
  ///           if (!equal(lhs->at(index), rhs->at(index), element_paths)) {
  ///             return false;
  ///           }
  ///         }
  ///         if (lhs->size() == rhs->size()) {
  ///           return true;
  ///         }
  ///         // If the array length is mismatched, don't report it immediately.
  ///         // Instead, defer the failure until we visit all children.
  ///         //
  ///         // This is for human readability. For example, say we have two sequences
  ///         //
  ///         //    (1)     a b c d e f g h i j k l m
  ///         //    (2)     a b c d e g h i j k l m
  ///         //
  ///         // If we directly report a mismatch at the end of the array right now,
  ///         // the user will see that array (1) has an element `m` at index 12 but array
  ///         (2)
  ///         // has no index 12 because it's too short:
  ///         //
  ///         //    (1)     a b c d e f g h i j k l m
  ///         //                                    ^error here
  ///         //    (2)     a b c d e g h i j k l m
  ///         //                                    ^ error here
  ///         //
  ///         // This is not very helpful. Instead, if we defer reporting this mismatch
  ///         // until all elements
  ///         // are fully visited, we can be much more helpful with pointing out the
  ///         // location:
  ///         //
  ///         //    (1)     a b c d e f g h i j k l m
  ///         //                      ^
  ///         //                   error here
  ///         //
  ///         //    (2)     a b c d e g h i j k l m
  ///         //                      ^
  ///         //                  error here
  ///         if (equal->IsFailDeferralEnabled()) {
  ///           if (lhs->size() > min_size) {
  ///             equal->DeferFail({array_paths->lhs_path->ArrayIndex(min_size),
  ///                               array_paths->rhs_path->MissingArrayElement(min_size)});
  ///           } else {
  ///             equal->DeferFail({array_paths->lhs_path->MissingArrayElement(min_size),
  ///                               array_paths->rhs_path->ArrayIndex(min_size)});
  ///           }
  ///           // Can return `true` pretending that everything is good since we have
  ///           // deferred the failure. return true;
  ///         }
  ///         return false;
  ///       }
  ///     };
  ///     TVM_REGISTER_REFLECTION_VTABLE(ArrayNode, ArrayNodeTrait)
  ///         .set_creator([](const std::string&) -> ObjectPtr<Object> {
  ///           return ::tvm::runtime::make_object<ArrayNode>();
  ///         });
  ///
  /// The last two vectors are registered by using `set_creator` and `set_repr_bytes`
  /// functions in `ReflectionVTable::Registry`, for example:
  ///     TVM_REGISTER_REFLECTION_VTABLE(ArrayNode, ArrayNodeTrait)
  ///         .set_creator([](const std::string&) -> ObjectPtr<Object> {
  ///           return ::tvm::runtime::make_object<ArrayNode>();
  ///         });
  ///
  /// Usage:
  ///     /*!
  ///      * \brief Create an initial object using default constructor
  ///      *        by type_key and global key.
  ///      *
  ///      * \param type_key The type key of the object.
  ///      * \param repr_bytes Bytes representation of the object if any.
  ///      */
  ///     TVM_DLL ObjectPtr<Object> CreateInitObject(const std::string& type_key,
  ///                                                const std::string& repr_bytes = "")
  ///                                                const;
  ///
  ///     /*!
  ///      * \brief Create an object by giving kwargs about its fields.
  ///      *
  ///      * \param type_key The type key.
  ///      * \param kwargs the arguments in format key1, value1, ..., key_n, value_n.
  ///      * \return The created object.
  ///      */
  ///     TVM_DLL ObjectRef CreateObject(const std::string& type_key,
  ///                                    const runtime::TVMArgs& kwargs);
  ///
  ///     /*!
  ///      * \brief Create an object by giving kwargs about its fields.
  ///      *
  ///      * \param type_key The type key.
  ///      * \param kwargs The field arguments.
  ///      * \return The created object.
  ///      */
  ///     TVM_DLL ObjectRef CreateObject(const std::string& type_key,
  ///                                    const Map<String, ObjectRef>& kwargs);
  ///
  ///     /*!
  ///      * \brief Get an field object by the attr name.
  ///      * \param self The pointer to the object.
  ///      * \param attr_name The name of the field.
  ///      * \return The corresponding attribute value.
  ///      * \note This function will throw an exception if the object does not contain
  ///      *       the field.
  ///      */
  ///     TVM_DLL runtime::TVMRetValue GetAttr(Object* self,
  ///                                          const String& attr_name) const;
  ///
  ///     /*!
  ///      * \brief List all the fields in the object.
  ///      * \return All the fields.
  ///      */
  ///     TVM_DLL std::vector<std::string> ListAttrNames(Object* self) const;
  ///
  ///     /*! \return The global singleton. */
  ///     TVM_DLL static ReflectionVTable* Global();
  ///
  ReflectionVTable *vtable = ReflectionVTable::Global();

  /// Test PrimExpr

  PrimExpr start = 4;
  std::vector<std::string> attrnames =
      vtable->ListAttrNames(const_cast<PrimExprNode *>(start.get()));
  LOG_PRINT_VAR(attrnames);

  MyIRSerializer serializer;
  vtable->VisitAttrs(const_cast<PrimExprNode *>(start.get()), &serializer);

  std::string reprbytes = "tmp";
  vtable->GetReprBytes(const_cast<PrimExprNode *>(start.get()), &reprbytes);
  LOG_PRINT_VAR(reprbytes);

  /// Test IterVar

  PrimExpr end = 8;
  Range range{start, end};
  Var var{"x", DataType::Int(32)};
  IterVar itervar{range, var, IterVarType::kThreadIndex, "threadidx.x"};

  attrnames = vtable->ListAttrNames(const_cast<IterVarNode *>(itervar.get()));
  LOG_PRINT_VAR(attrnames);
  Range retvalrange = vtable->GetAttr(const_cast<IterVarNode *>(itervar.get()), "dom");
  LOG_PRINT_VAR(retvalrange);

  vtable->VisitAttrs(const_cast<IterVarNode *>(itervar.get()), &serializer);

  vtable->GetReprBytes(const_cast<IterVarNode *>(itervar.get()), &reprbytes);
  LOG_PRINT_VAR(reprbytes);

  /// New object

  ObjectPtr<Object> x = vtable->CreateInitObject(itervar->_type_key, "tmpitervar");
  vtable->VisitAttrs(const_cast<IterVarNode *>(IterVar(x).get()), &serializer);
}

}  // namespace reflection_test

REGISTER_TEST_SUITE(reflection_test::AttrVisitorTest,
                    node_reflection_test_AttrVisitorTest);
REGISTER_TEST_SUITE(reflection_test::ReflectionVTableTest,
                    node_reflection_test_ReflectionVTableTest);
