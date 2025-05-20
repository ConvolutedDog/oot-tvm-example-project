#include "tvm/te/operation.h"
#include <tvm/te/tensor.h>

namespace operation_test {

using tvm::te::BaseComputeOpNode;
using tvm::te::compute;
using tvm::te::ComputeOp;
using tvm::te::ComputeOpNode;
using tvm::te::ExternOp;
using tvm::te::ExternOpNode;
using tvm::te::FBatchCompute;
using tvm::te::FCompute;
using tvm::te::Operation;
using tvm::te::OperationNode;
using tvm::te::placeholder;
using tvm::te::PlaceholderOp;
using tvm::te::PlaceholderOpNode;
using tvm::te::reduce_axis;
using tvm::te::scan;
using tvm::te::ScanOp;
using tvm::te::ScanOpNode;
using tvm::te::TensorDom;
using tvm::te::thread_axis;
using tvm::te::var;

using tvm::DataType;
using tvm::PrimExpr;
using tvm::runtime::Array;
using tvm::runtime::Map;
using tvm::runtime::String;
using tvm::te::Tensor;
using tvm::tir::Buffer;
using tvm::tir::IterVar;
using tvm::tir::Stmt;

void TePlaceholderOpTest();
void TeComputeOpTest();
void TeScanOpTest();
void TeExternOpTest();
void TeOtherFuncTest();

}  // namespace operation_test
