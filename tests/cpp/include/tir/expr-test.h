#include "tvm/ir/expr.h"
#include "tvm/te/operation.h"
#include "tvm/te/tensor.h"
#include "tvm/tir/expr.h"

namespace expr_test {

using tvm::tir::Add;
using tvm::tir::And;
using tvm::tir::Cast;
using tvm::tir::Div;
using tvm::tir::EQ;
using tvm::tir::FloorDiv;
using tvm::tir::FloorMod;
using tvm::tir::GE;
using tvm::tir::GT;
using tvm::tir::LE;
using tvm::tir::LT;
using tvm::tir::Max;
using tvm::tir::Min;
using tvm::tir::Mod;
using tvm::tir::Mul;
using tvm::tir::NE;
using tvm::tir::Not;
using tvm::tir::Or;
using tvm::tir::Select;
using tvm::tir::StringImm;
using tvm::tir::Sub;

using tvm::tir::Broadcast;
using tvm::tir::Buffer;
using tvm::tir::BufferLoad;
using tvm::tir::BufferType;
using tvm::tir::Call;
using tvm::tir::CommReducer;
using tvm::tir::Let;
using tvm::tir::ProducerLoad;
using tvm::tir::Ramp;
using tvm::tir::Reduce;
using tvm::tir::Shuffle;
using tvm::tir::Var;

using tvm::tir::as_unordered_map;

using tvm::DataType;
using tvm::IntImm;
using tvm::PointerType;
using tvm::PrimExpr;
using tvm::PrimType;
using tvm::Span;

using tvm::runtime::Array;
using tvm::runtime::DataType;
using tvm::runtime::String;

using tvm::te::PlaceholderOp;
using tvm::te::Tensor;

void TirExprTest();
void BufferLoadTest();
void ProducerLoadTest();
void RampTest();
void BroadcastTest();
void LetTest();
void TirCallTest();
void ShuffleTest();
void CommReducerTest();
void ReduceTest();

}  // namespace expr_test
