#include "ir/attrs-test.h"
#include "test-func-registry.h"
#include "tvm/ir/op.h"
#include "tvm/relax/attrs/nn.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

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

}  // namespace attrs_test

REGISTER_TEST_SUITE(attrs_test::IrAttrUtilsTests, ir_attrs_test_IrAttrUtilsTests);
REGISTER_TEST_SUITE(attrs_test::IrAttrFieldInfoTest, ir_attrs_test_IrAttrFieldInfoTest);
REGISTER_TEST_SUITE(attrs_test::IrAttrsTest, ir_attrs_test_IrAttrsTest);
REGISTER_TEST_SUITE(attrs_test::IrDictAttrsTest, ir_attrs_test_IrDictAttrsTest);
REGISTER_TEST_SUITE(attrs_test::IrAttrBriefTest, ir_attrs_test_IrAttrBriefTest);
