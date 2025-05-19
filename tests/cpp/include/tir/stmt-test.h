#include "tvm/te/tensor.h"
#include "tvm/tir/stmt.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/string.h>
#include <tvm/te/operation.h>

namespace stmt_test {

using tvm::tir::Allocate;
using tvm::tir::AllocateConst;
using tvm::tir::AssertStmt;
using tvm::tir::AttrStmt;
using tvm::tir::Block;
using tvm::tir::BlockRealize;
using tvm::tir::BufferRealize;
using tvm::tir::BufferRegion;
using tvm::tir::BufferStore;
using tvm::tir::DeclBuffer;
using tvm::tir::Evaluate;
using tvm::tir::For;
using tvm::tir::IfThenElse;
using tvm::tir::LetStmt;
using tvm::tir::MatchBufferRegion;
using tvm::tir::Prefetch;
using tvm::tir::ProducerRealize;
using tvm::tir::ProducerStore;
using tvm::tir::SeqStmt;
using tvm::tir::Stmt;
using tvm::tir::TypeAnnotation;
using tvm::tir::While;

using tvm::DataType;
using tvm::IntImm;
using tvm::PointerType;
using tvm::PrimExpr;
using tvm::PrimType;
using tvm::Span;
using tvm::runtime::Array;
using tvm::runtime::String;
using tvm::te::PlaceholderOp;
using tvm::te::Tensor;
using tvm::tir::Broadcast;
using tvm::tir::Buffer;
using tvm::tir::BufferType;
using tvm::tir::ForKind;
using tvm::tir::IterVar;
using tvm::tir::Var;

void BufferStoreTest();
void ProducerStoreTest();
void AllocateTest();
void IfThenElseTest();
void ForTest();
void PrefetchTest();
void TypeAnnotationTest();

}  // namespace stmt_test
