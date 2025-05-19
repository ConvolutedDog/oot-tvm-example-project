#include "tvm/relax/type.h"
#include "tvm/te/operation.h"
#include "tvm/te/tensor.h"
#include "tvm/tir/function.h"

namespace function_test {

using tvm::tir::PrimFunc;
using tvm::tir::Specialize;
using tvm::tir::TensorIntrin;

using tvm::DataType;
using tvm::DictAttrs;
using tvm::PrimExpr;
using tvm::Type;
using tvm::relax::TensorType;
using tvm::runtime::Array;
using tvm::runtime::Map;
using tvm::te::PlaceholderOp;
using tvm::te::Tensor;
using tvm::tir::Broadcast;
using tvm::tir::Buffer;
using tvm::tir::ProducerLoad;
using tvm::tir::ProducerStore;
using tvm::tir::Stmt;
using tvm::tir::Var;

void TirPrimFuncTest();
void TirTensorIntrinTest();
void TirSpecialize();

}  // namespace function_test
