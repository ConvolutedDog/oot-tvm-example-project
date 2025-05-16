#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"

namespace buffer_test {

using tvm::AttrVisitor;
using tvm::IntImm;
using tvm::PrimExpr;
using tvm::PrimType;
using tvm::Span;

using tvm::tir::Broadcast;
using tvm::tir::Buffer;
using tvm::tir::BufferNode;
using tvm::tir::BufferType;
using tvm::tir::BufferWithOffsetAlignment;
using tvm::tir::DataProducer;
using tvm::tir::decl_buffer;
using tvm::tir::Stmt;
using tvm::tir::Var;

using tvm::PointerType;

using tvm::runtime::Array;
using tvm::runtime::DataType;
using tvm::runtime::String;

void BufferTest();

/// @brief `DataProducerNode` is a virtual class that represents a data
/// producer. It will be used in `tvm::te::Tensor` and we will test it
/// when we use `tvm::te::Tensor`.
void DataProducerTest();

}  // namespace buffer_test
