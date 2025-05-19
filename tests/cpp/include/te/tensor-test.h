#include "tvm/te/operation.h"
#include "tvm/te/tensor.h"

namespace tensor_test {

using tvm::arith::IntSet;
using tvm::te::Operation;
using tvm::te::OperationNode;
using tvm::te::Tensor;
using tvm::te::TensorNode;

using tvm::DataType;
using tvm::PrimExpr;
using tvm::runtime::Array;
using tvm::tir::IterVar;
using tvm::tir::IterVarType;
using tvm::tir::Var;

using tvm::te::ComputeOp;

void TeTensorTest();

}  // namespace tensor_test
