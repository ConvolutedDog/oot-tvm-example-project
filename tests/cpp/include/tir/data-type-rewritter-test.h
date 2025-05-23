#include "tvm/tir/builtin.h"
#include "tvm/tir/data_type_rewriter.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"
#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace data_type_rewriter_test {

using tvm::tir::DataTypeLegalizer;
using tvm::tir::IndexDataTypeNormalizer;
using tvm::tir::IndexDataTypeRewriter;

using tvm::DataType;
using tvm::Integer;
using tvm::IntImm;
using tvm::make_object;
using tvm::PrimExpr;
using tvm::Range;
using tvm::runtime::Downcast;
using tvm::tir::Block;
using tvm::tir::BlockNode;
using tvm::tir::BlockRealize;
using tvm::tir::BlockRealizeNode;
using tvm::tir::Call;
using tvm::tir::const_true;
using tvm::tir::Evaluate;
using tvm::tir::For;
using tvm::tir::ForKind;
using tvm::tir::ForNode;
using tvm::tir::IfThenElse;
using tvm::tir::IterVar;
using tvm::tir::IterVarNode;
using tvm::tir::IterVarType;
using tvm::tir::Ramp;
using tvm::tir::RampNode;
using tvm::tir::Select;
using tvm::tir::SelectNode;
using tvm::tir::Stmt;
using tvm::tir::Var;
using tvm::tir::builtin::if_then_else;

void TirDataTypeLegalizerTest();
void TirIndexDataTypeRewriterTest();
void TirIndexDataTypeNormalizerTest();

}  // namespace data_type_rewriter_test
