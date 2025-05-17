#include "tvm/ir/expr.h"
#include "tvm/node/reflection.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/object.h"
#include "tvm/tir/var.h"

namespace reflection_test {

using tvm::AttrVisitor;
using tvm::DataType;
using tvm::PrimExpr;
using tvm::PrimExprNode;
using tvm::Range;
using tvm::ReflectionVTable;
using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::tir::IterVar;
using tvm::tir::IterVarNode;
using tvm::tir::IterVarType;
using tvm::tir::Var;

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

void NodeAttrVisitorTest();
void NodeReflectionVTableTest();

}  // namespace reflection_test
