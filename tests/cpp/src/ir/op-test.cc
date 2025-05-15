#include "ir/op-test.h"
#include "dlpack/dlpack.h"
#include "test-func-registry.h"

namespace op_test {

std::ostream &operator<<(std::ostream &os, const tvm::runtime::NDArray &arr) {  // NOLINT
  os << "Not implemented.\n";
  return os;
}

/// @brief Not safe.
std::ostream &operator<<(std::ostream &os, const TVMValue &tv) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&tv);
  for (size_t i = 0; i < sizeof(TVMValue); ++i) {
    os << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i])
       << " ";
  }
  return os;
}

void OpNodeTest() {
  LOG_SPLIT_LINE("OpNodeTest");

  PrimType arg1{DataType{DLDataType{DLDataTypeCode::kDLFloat, 32, 1}}};
  PointerType arg2{arg1};
  TupleType ret{
      {arg1, arg2}
  };
  FuncType functype{
      {arg1, arg2},
      ret
  };

  tvm::ObjectPtr<AttrFieldInfoNode> fieldinfo1 = tvm::make_object<AttrFieldInfoNode>();
  fieldinfo1->name = "arg1-fieldinfo";
  fieldinfo1->type_info = arg1.get()->_type_key;
  fieldinfo1->description = "arg1-fieldinfo";
  AttrFieldInfo field1 = AttrFieldInfo(fieldinfo1);

  tvm::ObjectPtr<AttrFieldInfoNode> fieldinfo2 = tvm::make_object<AttrFieldInfoNode>();
  fieldinfo2->name = "arg2-fieldinfo";
  fieldinfo2->type_info = arg2.get()->_type_key;
  fieldinfo2->description = "arg2-fieldinfo";
  AttrFieldInfo field2 = AttrFieldInfo(fieldinfo2);

  OpNode opnode;
  opnode.name = "test";
  opnode.op_type = functype;
  opnode.description = "This is a test op.";
  opnode.arguments = {field1, field2};

  MyIRSerializer serializer;
  opnode.VisitAttrs(&serializer);
}

/// @brief `Op` is registered by `TVM_REGISTER_OP` macro. For example,
///   TVM_REGISTER_OP("relax.nn.conv2d")
///       .set_num_inputs(2)
///       .add_argument("data", "Tensor", "The input tensor.")
///       .add_argument("weight", "Tensor", "The weight tensor.")
///       .set_attrs_type<Conv2DAttrs>()
///       .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2d)
///       .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConv2d)
///       .set_attr<TMixedPrecisionPolicy>(
///           "TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
///       .set_attr<FInferMixedPrecision>(
///           "FInferMixedPrecision", InferMixedPrecisionConv2d)
///       .set_attr<Bool>("FPurity", Bool(true));
/// And the attrs of `Op` is registered by `TVM_DECLARE_ATTRS`, such as:
///   struct Conv2DAttrs : public tvm::AttrsNode<Conv2DAttrs> {
///       Array<IntImm> strides;
///       Array<IntImm> padding;
///       Array<IntImm> dilation;
///       int groups;
///       String data_layout;
///       String kernel_layout;
///       String out_layout;
///       DataType out_dtype;
///   TVM_DECLARE_ATTRS(Conv2DAttrs, "relax.attrs.Conv2DAttrs") {
///       TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
///       ...
///   }
///  };
///
/// In the above example, when we use function like `.set_attr<Bool>("FPurity",
/// Bool(true))` to define an attribute for a specific operator, we are actually
/// calling `OpRegEntry::set_attr(string attr_name, Bool& value, int plevel)` and
/// it returns an `OpRegEntry&` reference. Here, the `OpRegEntry` is a helper
/// structure to register operators. TVM will allocate an `OpRegEntry` object for each
/// operator, that is to say each `OpRegEntry` contains an `Op op_` operator and the
/// `OpNode` object can be accessed by `OpNode* get()`. `OpRegEntry` can also set
/// `description`, `name`, `type_info`, `arguments`, `num_inputs`, `attrs_type_key`
/// `attrs_type_index`, and `support_level` for the internal `OpNode` storage of this
/// operator. Also `OpRegEntry` will also set attributes for the operator.
///
/// The regstration of attributes is done by a registry named `AttrRegistry`. It is a
/// template class that takes two template arguments: `EntryType` and `KeyType`. The
/// first Tparam `EntryType` corresponds to the above `OpRegEntry` class. We have said
/// each `OpRegEntry` contains an `Op op_` operator. Here `AttrRegistry` is a singleton
/// class that contains a private `unordered_map<String, EntryType*> entry_map_` member,
/// and it stores a map from `String` to `EntryType*`. Here we refer to the defination
/// of `TVM_REGISTER_OP`:
///
///   #define TVM_REGISTER_OP(OpName)                                              \
///     static DMLC_ATTRIBUTE_UNUSED ::tvm::OpRegEntry& __make_##Op##__COUNTER__ = \
///       ::tvm::OpRegEntry::RegisterOrGet(OpName).set_name()
///
/// `OpRegEntry::RegisterOrGet(OpName)` will call `OpRegistry::Global()->RegisterOrGet`
///  where `OpRegistry = AttrRegistry<OpRegEntry, Op>` (In fact, `Op` is the `KeyType`
/// Tparam we discussed in the previous part and we will discuss it later). And here,
/// `OpRegistry::Global()` will access the singleton instance of `OpRegistry`. And
/// `OpRegistry::RegisterOrGet(OpName)` will find the op's corresponding `OpRegEntry`
/// instance from the `unordered_map<String, EntryType*> entry_map_` member and use this
/// instance to do the following `set_name()` setting. But the `OpName` may not be
/// inside the `entry_map_` member, at this point, the `OpRegEntry` instance will create
/// a new `EntryType*` instance and insert it into the `entry_map_` member. Also, the
/// `OpRegEntry` instance maintains a `vector<unique_ptr<EntryType>> entries_` member,
/// but this is only used to count the index to insert when a new `EntryType*` instance
/// is created.
///
/// Backward to the `KeyType`, we have already kown how to get the `EntryType*` instance
/// for an operator, but to achieve the goal of setting attributes, we also need to seek
/// the attributes for each operator. When we use the `TVM_REGISTER_OP` macro to define
/// attributes, we use `.set_attr<Bool>("FPurity", Bool(true))`. This will finally call
/// the `OpRegEntry::UpdateAttr(attr_name, value, plevel)` and the latter will call the
/// `OpRegistry::UpdateAttr(attr_name, op_, TVMRetValue(value), plevel)` to update the
/// registry. From the defination of `OpRegistry::UpdateAttr`, we can infer that
/// `KeyType` is `Op`. This function will find if the attribute that corresponds to
/// `attr_name` is inside the `OpRegistry::attrs_`.
///
/// There's a private `unordered_map<String, AttrRegistryMapContainerMap<Op>*> attrs_`
/// in `OpRegistry` (=`AttrRegistry<OpRegEntry, Op>`) to store the generic attribute
/// map. This map is indexed by the `Op` instance. In fact, the authors have declared
/// that `AttrRegistryMapContainerMap<Op>` is a generic attribute map. It has a private
/// member named `attr_name_`, which means that each attribute with a specific attribute
/// name has a single `AttrRegistryMapContainerMap<Op>` instance. It has also a private
/// `vector<pair<TVMRetValue, int>> data_` member, this is to store the attribute value
/// of the attribute that has a name equals to `attr_name_` of each operator. And the
/// index of the `pair` in `data_` is the program internal unique index of operator,
/// which is defined in `OpNode` class.
///
/// So almost all the registration of Op and its attributes is completed here.
void OpTest() {
  LOG_SPLIT_LINE("OpTest");

  /// @brief We can use `OpRegistry::ListAllNames()` to get all the operator names that
  /// are registered in the system.
  OpRegistry *opregistry = OpRegistry::Global();
  LOG_PRINT_VAR(opregistry->ListAllNames());

  /// @brief We can use `Op::Get()` to get the `Op` operator by its name.
  Op op = Op::Get("relax.nn.conv2d");

  MyIRSerializer serializer;
  const_cast<OpNode *>(op.get())->VisitAttrs(&serializer);
  LOG_PRINT_VAR(Op::HasAttrMap("FInferStructInfo"));
  LOG_PRINT_VAR(Op::HasAttrMap("FRelaxInferLayout"));
  LOG_PRINT_VAR(Op::HasAttrMap("TMixedPrecisionPolicy"));
  LOG_PRINT_VAR(Op::HasAttrMap("FInferMixedPrecision"));
  LOG_PRINT_VAR(Op::HasAttrMap("FPurity"));

  /// @brief Here, `Op::GetAttrMap` is used to get the generic attribute map. We have
  /// to say that `OpAttrMap<tvm::Bool>` inherits from `AttrRegistryMap<Op, tvm::Bool>`
  /// which is a map from `Op` to `tvm::Bool`. In fact, when we call `Op::GetAttrMap`,
  /// it actually will call `OpAttrMap<tvm::Bool>(Op::GetAttrMapContainer(key))`, where
  /// the `key` is the attribute name. And we have also discussed `OpRegistry::attrs_`
  /// stores the generic attribute map, so `Op::GetAttrMapContainer(key)` actually get
  /// `AttrRegistryMapContainerMap<Op>*` instance and initialize `OpAttrMap<tvm::Bool>`
  /// with it.
  ///
  /// So, `Op::GetAttrMap` access the generic attribute map, this map corresponds to an
  /// attribute named "FPurity" and it stores the `tvm::Bool` value for each operator.
  const OpAttrMap<tvm::Bool> &opattrmap = Op::GetAttrMap<tvm::Bool>("FPurity");
  /// @brief We can use `Op` operator to get the attribute value.
  LOG_PRINT_VAR(opattrmap.count(op));
  LOG_PRINT_VAR(opattrmap[op]);

  /// @brief Set a test operator.
  TVM_REGISTER_OP("testop").set_name();

  OpRegEntry &opregentry = OpRegEntry::RegisterOrGet("testop");
  opregentry = *(opregistry->Get("testop"));
  opregentry.describe("Description of testop");
  opregentry.set_num_inputs(4);
  opregentry.set_support_level(11);
  opregentry.add_argument("testa", "int", "desca");
  opregentry.add_argument("testb", "int", "descb");
  opregentry.set_attrs_type<Conv2DAttrs>();
  opregentry.set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy",
                                             MixedPrecisionPolicyKind::kAlways);
  opregentry.set_attr<tvm::Bool>("FPurity", tvm::Bool(true));

  Op testop = Op::Get("testop");
  const_cast<OpNode *>(testop.get())->VisitAttrs(&serializer);

  using OpRegistry = AttrRegistry<OpRegEntry, Op>;
  const AttrRegistryMapContainerMap<Op> &attrregmapcontainermap =
      opregistry->GetAttrMap("FPurity");
  LOG_PRINT_VAR(attrregmapcontainermap.count(testop));
  const tvm::runtime::TVMRetValue &retval = attrregmapcontainermap[testop];
}

}  // namespace op_test

REGISTER_TEST_SUITE(op_test::OpNodeTest, ir_op_test_OpNodeTest);
REGISTER_TEST_SUITE(op_test::OpTest, ir_op_test_OpTest);
