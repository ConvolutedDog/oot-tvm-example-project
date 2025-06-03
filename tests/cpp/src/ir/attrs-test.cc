#include "ir/attrs-test.h"
#include "test-func-registry.h"
#include "tvm/../../src/relax/op/tensor/linear_algebra.h"
#include "tvm/ir/op.h"
#include "tvm/relax/attrs/nn.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/op_attr_types.h"
#include "tvm/relax/struct_info.h"
#include "utils.h"
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>

namespace attrs_test {

void IrAttrUtilsTests() {
  LOG_SPLIT_LINE("IrAttrUtilsTests");

  LOG_PRINT_VAR(NullValue<AttrFieldInfo>());  // nullptr
  LOG_PRINT_VAR(NullValue<DataType>());       // void
  /// Not derived from DataType, but a ObjectRef.
  LOG_PRINT_VAR(NullValue<PointerType>());  // nullptr
}

void IrAttrFieldInfoTest() {
  LOG_SPLIT_LINE("IrAttrFieldInfoTest");

  MyIRSerializer serializer;
  AttrFieldInfo attrfieldinfo{make_object<AttrFieldInfoNode>()};
  AttrFieldInfoNode *attrfieldinfoptr =
      const_cast<AttrFieldInfoNode *>(attrfieldinfo.get());

  attrfieldinfoptr->name = "name";
  attrfieldinfoptr->type_info = "type_info";
  attrfieldinfoptr->description = "description";

  const_cast<AttrFieldInfoNode *>(attrfieldinfo.as<AttrFieldInfoNode>())
      ->VisitAttrs(&serializer);
}

void MyAttrNode::VisitNonDefaultAttrs(AttrVisitor *v) { v->Visit("__attr3__", &attr3); }

#define GEN_ATTR_FIELD_INFO(No, Name, Type_Info, Description)                            \
  AttrFieldInfo attrfieldinfo##No{make_object<AttrFieldInfoNode>()};                     \
  AttrFieldInfoNode *attrfieldinfoptr##No =                                              \
      const_cast<AttrFieldInfoNode *>(attrfieldinfo##No.get());                          \
  attrfieldinfoptr##No->name = Name;                                                     \
  attrfieldinfoptr##No->type_info = Type_Info;                                           \
  attrfieldinfoptr##No->description = Description;

tvm::runtime::Array<AttrFieldInfo> MyAttrNode::ListFieldInfo() const {
  tvm::runtime::Array<AttrFieldInfo> fields;

  GEN_ATTR_FIELD_INFO(1, "attr1", "std::string", "description of attr1");
  GEN_ATTR_FIELD_INFO(2, "attr2", "tvm::PrimExpr", "description of attr2");
  GEN_ATTR_FIELD_INFO(3, "attr3", "std::string", "description of attr3");

  fields.push_back(attrfieldinfo1);
  fields.push_back(attrfieldinfo2);
  fields.push_back(attrfieldinfo3);

  return fields;
}

MyAttr::MyAttr(std::string attr1, tvm::PrimExpr attr2, std::string attr3) {
  ObjectPtr<MyAttrNode> n = make_object<MyAttrNode>();
  n->attr1 = std::move(attr1);
  n->attr2 = std::move(attr2);
  n->attr3 = std::move(attr3);
  data_ = std::move(n);
}

void IrAttrsTest() {
  LOG_SPLIT_LINE("AttrsTest");

  MyIRSerializer serializer;

  MyAttr attr1{make_object<MyAttrNode>()};
  const_cast<MyAttrNode *>(attr1.as<MyAttrNode>())->VisitAttrs(&serializer);

  MyAttr attr2{"attr1", 2, "attr3"};
  const_cast<MyAttrNode *>(attr2.as<MyAttrNode>())->VisitAttrs(&serializer);
  const_cast<MyAttrNode *>(attr2.as<MyAttrNode>())->VisitNonDefaultAttrs(&serializer);

  LOG_PRINT_VAR("attr2.get()->PrintDocString(std::cout) START");
  attr2.get()->PrintDocString(std::cout);
  LOG_PRINT_VAR("attr2.get()->PrintDocString(std::cout) END");

  LOG_PRINT_VAR(attr2.get()->ListFieldInfo());
}

void IrDictAttrsTest() {
  LOG_SPLIT_LINE("IrDictAttrsTest");

  std::initializer_list<std::pair<String, ObjectRef>> init = {
      {"attr1", String("attr1") },
      {"attr2", tvm::PrimExpr(0)},
      {"attr3", String("attr3") },
  };

  Map<String, ObjectRef> map{init};

  DictAttrs dict(map);

  LOG_PRINT_VAR(dict.GetAttr<String>("attr1"));         // "attr1"
  LOG_PRINT_VAR(dict.GetAttr<tvm::PrimExpr>("attr2"));  // 0
  LOG_PRINT_VAR(dict.GetAttr<String>("attr3"));         // "attr3"

  LOG_PRINT_VAR(dict.HasNonzeroAttr("attr2"));  // 0

  LOG_PRINT_VAR(AttrsWithDefaultValues<DictAttrs>());      // {}
  LOG_PRINT_VAR(AttrsWithDefaultValues<MyAttr>());         // test.MyAttrNode(0x155f578)
  LOG_PRINT_VAR(AttrsWithDefaultValues<MyAttr>()->attr1);  // ""
  LOG_PRINT_VAR(AttrsWithDefaultValues<MyAttr>()->attr2);  // nullptr
  LOG_PRINT_VAR(AttrsWithDefaultValues<MyAttr>()->attr3);  // ""

  std::initializer_list<std::pair<String, ObjectRef>> newinit = {
      {"attr1", String("attr1-1")},
      {"attr2", tvm::PrimExpr(0) },
      {"attr3", String("attr3-1")},
  };
  LOG_PRINT_VAR(WithAttrs(dict, newinit));
  /// Output:
  ///   {"attr1": "attr1-1", "attr2": 0, "attr3": "attr3-1"}

  LOG_PRINT_VAR(WithAttr(dict, String("attr2"), tvm::PrimExpr(1)));
  /// Output:
  ///   {"attr1": "attr1", "attr2": 1, "attr3": "attr3"}

  LOG_PRINT_VAR(WithoutAttr(dict, "attr2"));
  /// Output:
  ///   {"attr1": "attr1", "attr3": "attr3"}
}

/// @brief Attributes is a mechanism for managing static attributes known at compile time
/// in deep learning models. These attributes are usually determined at model compilation
/// time (rather than runtime) and are used to guide optimization and code generation.
void IrAttrBriefTest() {
  LOG_PRINT_VAR("IrAttrBriefTest");

  tvm::Op op = tvm::Op::Get("relax.nn.conv2d");
  tvm::relax::Var x{"x", tvm::relax::ShapeStructInfo{4}};
  tvm::relax::Var y{"y", tvm::relax::ShapeStructInfo{4}};
  auto convattrs = make_object<tvm::relax::Conv2DAttrs>();

  using tvm::IntImm;
  convattrs->strides = {
      IntImm{DataType::Int(32), 2},
      IntImm{DataType::Int(32), 2}
  };
  convattrs->padding = {
      IntImm{DataType::Int(32), 1},
      IntImm{DataType::Int(32), 1}
  };
  convattrs->dilation = {
      IntImm{DataType::Int(32), 1},
      IntImm{DataType::Int(32), 1}
  };
  convattrs->groups = 1;
  convattrs->data_layout = "NCHW";
  convattrs->kernel_layout = "OIHW";
  convattrs->out_layout = "NCHW";
  convattrs->out_dtype = tvm::DataType::BFloat(16);
  tvm::relax::Call call{
      op, {x, y},
       Attrs(convattrs)
  };
  LOG_PRINT_VAR(call);
  /// Output:
  ///   R.nn.conv2d(x, y, strides=[2, 2], padding=[1, 1],
  ///               dilation=[1, 1], groups=1, data_layout="NCHW",
  ///               kernel_layout="OIHW", out_layout="NCHW", out_dtype="bfloat16")

  LOG_PRINT_VAR(call->attrs.as<tvm::relax::Conv2DAttrs>());
  call->attrs->PrintDocString(std::cout);
  LOG_PRINT_VAR(call->attrs->ListFieldInfo());
}

/// @brief Introduction to the additional attrs.
///
/// Attributes are a mechanism for managing static attributes known at compile time in
/// deep learning models. They are usually determined at model compilation time (rather
/// than runtime) and are used to guide optimization and code generation. We can simply
/// see them as a set of constant key-value pairs.
///
/// @note Here, we mainly introduce the additional attrs that are registered during the
/// operator registration. The type defination of these additional attrs is located at
/// `include/tvm/relax/op_attr_types.h`.
///
/// We take the `relax.matmul` Operator as an example:
/// @code{.cpp}
///   TVM_REGISTER_OP("relax.matmul")
///       .set_num_inputs(2)
///       .add_argument("x1", "Tensor", "The first input tensor.")
///       .add_argument("x2", "Tensor", "The second input tensor.")
///       .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMatmul)
///       .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy",
///                                        MixedPrecisionPolicyKind::kAlways)
///       .set_attr<FInferMixedPrecision>("FInferMixedPrecision",
///                                       InferMixedPrecisionMatmul)
///       .set_attr<Bool>("FPurity", Bool(true));
/// @endcode
///
/// TVM_REGISTER_OP("relax.matmul") can be expanded as follows:
/// @code{.cpp}
///   static __attribute__((unused)) ::tvm ::OpRegEntry& __make_Op3 =
///       ::tvm ::OpRegEntry ::RegisterOrGet("relax.matmul").set_name()
/// @endcode
///
/// `OpRegEntry` is a helper class that provides a convenient way to register an operator,
/// it contains the following fields:
///   1. std::string name;  // The name of the operator, "relax.matmul".
///   2. Op op_;            // The Op object.
/// Each `OpRegEntry` instance has only an operator and each operator will have a unique
/// `OpRegEntry` instance. The `OpRegEntry` instance is created when the operator is
/// registered by `TVM_REGISTER_OP("relax.matmul")`, it calls `OpRegEntry::RegisterOrGet`
/// method. And this method will call `OpRegistry::Global()->RegisterOrGet(name)` method
/// to register or get the corresponding `OpRegEntry` instance, where `OpRegistry` is:
///     using OpRegistry = AttrRegistry<OpRegEntry, Op>;
///
/// `OpRegistry` stores all `OpRegEntry` instances in a map called `std::unordered_map<
/// String, EntryType*> entry_map_` after all the operators have been registered, and the
/// key is the operator name. During the registration of an operator, the `OpRegistry`
/// will check if the operator has been registered before, if not, it will create a new
/// `OpRegEntry` instance and insert it into the `entry_map_`.
///
/// `OpRegEntry` has all the methods that can fill in the `OpNode`'s fields:
///   1. inline OpRegEntry& set_name();
///      => String OpNode::name
///   2. inline OpRegEntry& describe(const std::string& descr);
///      => String OpNode::description
///   3. inline OpRegEntry& add_argument(const std::string& name, const std::string& type,
///                                      const std::string& description);
///      => Array<AttrFieldInfo> OpNode::arguments
///   4. template <typename AttrsType> inline OpRegEntry& set_attrs_type();
///      => String OpNode::attrs_type_key and uint32_t OpNode::attrs_type_index
///   5. inline OpRegEntry& set_attrs_type_key(const String& key);
///      => String OpNode::attrs_type_key and uint32_t OpNode::attrs_type_index
///   6. inline OpRegEntry& set_num_inputs(int32_t n);
///      => int32_t OpNode::num_inputs
///   7. inline OpRegEntry& set_support_level(int32_t level);
///      => int32_t OpNode::support_level
/// And the `OpNode::index_` is the internal unique index of operator, it is assigned when
/// calling `OpRegEntry::RegisterOrGet`.
///
/// The `set_attr` method is used to set additional attributes of an operator. In the
/// "relax.matmul" example:
/// @code{.cpp}
///     using FInferStructInfo =
///       runtime::TypedPackedFunc<StructInfo(const Call& call, const BlockBuilder& ctx)>;
///     using TMixedPrecisionPolicy = int;
///     using FInferMixedPrecision =
///       runtime::TypedPackedFunc<Call(const Call& call_node, const DataType&
///       out_dtype)>;
/// @endcode
/// It use `set_attr` to register four additional attributes for the operator.
///
/// When calling the `set_attr` method, it calls `OpRegistry::Global()->UpdateAttr`
/// method. There is a private member in `OpRegistry`:
///     std::unordered_map<String, unique_ptr<AttrRegistryMapContainerMap<Op>>> attrs_
/// It stores the additional attributes for all of the operators. First the key of
/// `attrs_` is the additional attribute's name, and it corresponds to the unique pointer
/// that points the container map which stores the value of this additional attribute for
/// all of the operator. The container map has a private member:
///     std::vector<std::pair<runtime::TVMRetValue, int>> data_
/// It is indexed by the `OpNode::index_` --- which we have mentioned before --- and the
/// items it stores is the value and support_level of this additional attribute.
/// `OpRegistry::Global()->UpdateAttr` will check if this additional attribute has been
/// registered, if not it will regiatered it. And then it will check if there is value for
/// the current operator, if not it will add this value and support_level to the container
/// map, and if there is value for the current operator, it will update the value and
/// support_level.
///
/// @note Supplementary details of the normal attribute.
///
/// For the normal attribute, TVM provides a function to return the Call node with the
/// normal attribute called `MatMulAttrs`:
/// @code{.cpp}
///   struct MatmulAttrs : public tvm::AttrsNode<MatmulAttrs> {
///     DataType out_dtype;
///
///     TVM_DECLARE_ATTRS(MatmulAttrs, "relax.attrs.MatmulAttrs") {
///       TVM_ATTR_FIELD(out_dtype).describe("The data type of the output tensor");
///     }
///   };  // struct MatmulAttrs
///
///   TVM_REGISTER_NODE_TYPE(MatmulAttrs);
///
///   Expr matmul(Expr x1, Expr x2, DataType out_dtype) {
///     ObjectPtr<MatmulAttrs> attrs = make_object<MatmulAttrs>();
///     attrs->out_dtype = out_dtype;
///
///     static const Op& op = Op::Get("relax.matmul");
///     return Call(op, {std::move(x1), std::move(x2)}, Attrs(attrs), {});
///   }
/// @endcode
void IrAllAttrsIntroTest() {
  LOG_SPLIT_LINE("IrAllAttrsIntroTest");

  const tvm::Op &op = GetOpByStringName("relax.matmul");
  LOG_PRINT_VAR(op);
  /// Output:
  ///   op: Op(relax.matmul)

  LOG_PRINT_VAR(GetAdditioanlAttrValue<tvm::Bool>("relax.matmul", "FPurity"));
  /// Output:
  ///   GetAdditioanlAttrValue<tvm::Bool>("relax.matmul", "FPurity"): T.bool(True)

  LOG_PRINT_VAR(GetAdditioanlAttrValue<tvm::relax::FInferStructInfo>("relax.matmul",
                                                                     "FInferStructInfo"));
  /// Output:
  ///   GetAdditioanlAttrValue<tvm::relax::FInferStructInfo>(
  ///       "relax.matmul", "FInferStructInfo"): runtime.PackedFunc(0x12a0960)

  tvm::RelaxExpr expr = tvm::relax::matmul(tvm::GlobalVar{"a"}, tvm::GlobalVar{"b"},
                                           tvm::DataType::Float(32));
  tvm::Attrs attrs = GetNormalAttrValue(expr);
  LOG_PRINT_VAR(attrs);
  /// Output:
  ///   attrs: relax.attrs.MatmulAttrs(0x16cb9a8)
}

}  // namespace attrs_test

REGISTER_TEST_SUITE(attrs_test::IrAttrUtilsTests, ir_attrs_test_IrAttrUtilsTests);
REGISTER_TEST_SUITE(attrs_test::IrAttrFieldInfoTest, ir_attrs_test_IrAttrFieldInfoTest);
REGISTER_TEST_SUITE(attrs_test::IrAttrsTest, ir_attrs_test_IrAttrsTest);
REGISTER_TEST_SUITE(attrs_test::IrDictAttrsTest, ir_attrs_test_IrDictAttrsTest);
REGISTER_TEST_SUITE(attrs_test::IrAttrBriefTest, ir_attrs_test_IrAttrBriefTest);
REGISTER_TEST_SUITE(attrs_test::IrAllAttrsIntroTest, ir_attrs_test_IrAllAttrsIntroTest);
