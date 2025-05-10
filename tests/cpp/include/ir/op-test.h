#include "dlpack/dlpack.h"
#include "tvm/ir/attrs.h"
#include "tvm/ir/expr.h"
#include "tvm/ir/op.h"
#include "tvm/ir/type.h"
#include "tvm/relax/op_attr_types.h"
#include "tvm/runtime/data_type.h"
#include "tvm/../../src/node/attr_registry.h"
#include "tvm/relax/attrs/nn.h"
#include "tvm/../../src/relax/transform/infer_amp_utils.h"
#include "tvm/node/attr_registry_map.h"

namespace op_test {

using tvm::Op;
using tvm::OpNode;
using tvm::OpRegEntry;
using tvm::OpAttrMap;

using tvm::FuncType;
using tvm::FuncTypeNode;
using tvm::PointerType;
using tvm::PointerTypeNode;
using tvm::PrimType;
using tvm::PrimTypeNode;
using tvm::TupleType;
using tvm::TupleTypeNode;
using tvm::Type;
using tvm::TypeNode;

using tvm::runtime::DataType;

using tvm::AttrFieldInfo;
using tvm::AttrFieldInfoNode;

using tvm::AttrRegistry;
using OpRegistry = AttrRegistry<OpRegEntry, Op>;

using tvm::relax::Conv2DAttrs;
using tvm::relax::FInferStructInfo;
using tvm::relax::MixedPrecisionPolicyKind;
using tvm::relax::TMixedPrecisionPolicy;

using tvm::AttrRegistryMapContainerMap;

using tvm::AttrVisitor;

std::ostream &operator<<(std::ostream &os, const tvm::runtime::NDArray &arr);

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

void OpNodeTest();
void OpTest();

}  // namespace op_test

void OpNodeTest();
void OpTest();
