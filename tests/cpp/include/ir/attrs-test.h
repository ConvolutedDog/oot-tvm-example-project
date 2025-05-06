#include "tvm/ir/attrs.h"
#include "tvm/ir/type.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/memory.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>

namespace attrs_test {

using tvm::AttrFieldInfo;
using tvm::AttrFieldInfoNode;
using tvm::Attrs;
using tvm::AttrsWithDefaultValues;
using tvm::BaseAttrsNode;
using tvm::DictAttrs;
using tvm::DictAttrsNode;

using tvm::AttrVisitor;
using tvm::NullValue;
using tvm::String;
using tvm::runtime::make_object;
using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::ObjectRef;

using tvm::DataType;
using tvm::PointerType;

using tvm::runtime::Map;
using tvm::runtime::String;

class MyIRSerializer : public AttrVisitor {
  void Visit(const char *key, double *value) override {
    std::cout << " double:             " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, int64_t *value) override {
    std::cout << " int64_t:            " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, uint64_t *value) override {
    std::cout << " uint64_t:           " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, int *value) override {
    std::cout << " int:                " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, bool *value) override {
    std::cout << " bool:               " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, std::string *value) override {
    std::cout << " std::string:        " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, void **value) override {
    std::cout << " void:               " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, DataType *value) override {
    std::cout << " DataType:           " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, tvm::runtime::NDArray *value) override {
    std::cout << " runtime::NDArray:   " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, tvm::runtime::ObjectRef *value) override {
    std::cout << " runtime::ObjectRef: " << key << "=" << *value << ";\n";
  }
};

class MyAttrNode : public BaseAttrsNode {
public:
  std::string attr1;
  tvm::PrimExpr attr2;
  std::string attr3;

public:
  ~MyAttrNode() = default;
  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("attr1", &attr1);
    v->Visit("attr2", &attr2);
    v->Visit("attr3", &attr3);
  }
  void VisitNonDefaultAttrs(AttrVisitor *v) final;
  tvm::runtime::Array<AttrFieldInfo> ListFieldInfo() const final;
  // NOLINTNEXTLINE(readability-identifier-naming)
  void InitByPackedArgs(const TVMArgs &kwargs, bool allow_unknown = false) final {};

public:
  static constexpr const uint32_t _type_index = tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.MyAttrNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(MyAttrNode, BaseAttrsNode);
};

class MyAttr : public Attrs {
public:
  MyAttr(std::string attr1, tvm::PrimExpr attr2, std::string attr3);
  TVM_DEFINE_OBJECT_REF_METHODS(MyAttr, Attrs, MyAttrNode);
};

void AttrUtilsTests();
void AttrFieldInfoTest();
void AttrsTest();
void DictAttrsTest();

}  // namespace attrs_test

void AttrUtilsTests();
void AttrFieldInfoTest();
void AttrsTest();
void DictAttrsTest();
