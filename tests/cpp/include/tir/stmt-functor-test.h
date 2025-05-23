#include "tvm/tir/stmt_functor.h"

namespace stmt_functor_test {

using tvm::tir::StmtExprMutator;
using tvm::tir::StmtExprVisitor;
using tvm::tir::StmtFunctor;
using tvm::tir::StmtMutator;
using tvm::tir::StmtVisitor;

using tvm::tir::AllocateConstNode;
using tvm::tir::AllocateNode;
using tvm::tir::AssertStmtNode;
using tvm::tir::AttrStmtNode;
using tvm::tir::BlockNode;
using tvm::tir::BlockRealizeNode;
using tvm::tir::BufferRealizeNode;
using tvm::tir::BufferStoreNode;
using tvm::tir::DeclBufferNode;
using tvm::tir::EvaluateNode;
using tvm::tir::ForNode;
using tvm::tir::IfThenElseNode;
using tvm::tir::LetStmtNode;
using tvm::tir::PrefetchNode;
using tvm::tir::ProducerRealizeNode;
using tvm::tir::ProducerStoreNode;
using tvm::tir::SeqStmtNode;
using tvm::tir::WhileNode;

using tvm::NodeFunctor;
using tvm::PrimExpr;
using tvm::Range;
using tvm::runtime::Array;
using tvm::runtime::Map;
using tvm::runtime::ObjectPtr;
using tvm::runtime::String;
using tvm::tir::Evaluate;
using tvm::tir::ExprMutator;
using tvm::tir::LetStmt;
using tvm::tir::PrimFunc;
using tvm::tir::Stmt;
using tvm::tir::StmtMutator;
using tvm::tir::Var;

void TirStmtFunctorTest();
void TirOtherVisitorMutatorTest();

}  // namespace stmt_functor_test
