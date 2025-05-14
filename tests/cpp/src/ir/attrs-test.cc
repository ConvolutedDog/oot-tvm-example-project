#include "ir/attrs-test.h"
#include "test-func-registry.h"

namespace attrs_test {

void AttrUtilsTests() {
  LOG_SPLIT_LINE("AttrUtilsTests");

  LOG_PRINT_VAR(NullValue<AttrFieldInfo>());
  LOG_PRINT_VAR(NullValue<DataType>());
  /// Not derived from DataType, but a ObjectRef.
  LOG_PRINT_VAR(NullValue<PointerType>());
}

void AttrFieldInfoTest() {
  LOG_SPLIT_LINE("AttrFieldInfoTest");

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

void AttrsTest() {
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

void DictAttrsTest() {
  LOG_SPLIT_LINE("DictAttrsTest");

  std::initializer_list<std::pair<String, ObjectRef>> init = {
      {"attr1", String("attr1") },
      {"attr2", tvm::PrimExpr(0)},
      {"attr3", String("attr3") },
  };

  Map<String, ObjectRef> map{init};

  DictAttrs dict(map);

  LOG_PRINT_VAR(dict.GetAttr<String>("attr1"));
  LOG_PRINT_VAR(dict.GetAttr<tvm::PrimExpr>("attr2"));
  LOG_PRINT_VAR(dict.GetAttr<String>("attr3"));

  LOG_PRINT_VAR(dict.HasNonzeroAttr("attr2"));

  LOG_PRINT_VAR(AttrsWithDefaultValues<DictAttrs>());

  std::initializer_list<std::pair<String, ObjectRef>> newinit = {
      {"attr1", String("attr1-1")},
      {"attr2", tvm::PrimExpr(0) },
      {"attr3", String("attr3-1")},
  };
  LOG_PRINT_VAR(WithAttrs(dict, newinit));
  LOG_PRINT_VAR(WithoutAttr(dict, "attr2"));
}

}  // namespace attrs_test

void AttrUtilsTests() { attrs_test::AttrUtilsTests(); }
void AttrFieldInfoTest() { attrs_test::AttrFieldInfoTest(); }
void AttrsTest() { attrs_test::AttrsTest(); }
void DictAttrsTest() { attrs_test::DictAttrsTest(); }

namespace {

REGISTER_TEST_SUITE(AttrUtilsTests);
REGISTER_TEST_SUITE(AttrFieldInfoTest);
REGISTER_TEST_SUITE(AttrsTest);
REGISTER_TEST_SUITE(DictAttrsTest);

}  // namespace
