#include "tvm/tir/function.h"
#include "tvm/relax/type.h"
#include "tvm/te/tensor.h"
#include "tvm/te/operation.h"

namespace function_test {

using tvm::tir::PrimFunc;
using tvm::tir::TensorIntrin;
using tvm::tir::Specialize;

using tvm::tir::Var;
using tvm::tir::Stmt;
using tvm::Type;
using tvm::runtime::Map;
using tvm::tir::Buffer;
using tvm::DictAttrs;
using tvm::DataType;
using tvm::tir::Stmt;
using tvm::te::Tensor;
using tvm::relax::TensorType;
using tvm::runtime::Array;
using tvm::PrimExpr;
using tvm::te::PlaceholderOp;
using tvm::tir::ProducerStore;
using tvm::tir::Broadcast;
using tvm::tir::ProducerLoad;

void TirPrimFuncTest();
void TirTensorIntrinTest();
void TirSpecialize();

}
