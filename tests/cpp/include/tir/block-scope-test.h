#include "tvm/tir/block_scope.h"

namespace block_scope_test {

using tvm::tir::StmtSRef;
using tvm::tir::SRefTreeCreator;
using tvm::tir::DepKind;
using tvm::tir::Dependency;
using tvm::tir::BlockScope;

using tvm::tir::StmtNode;
using tvm::IRModule;
using tvm::tir::ForNode;
using tvm::tir::BlockRealizeNode;
using tvm::tir::SeqStmtNode;
using tvm::runtime::Array;

void TirStmtSRefTest();
void TirSRefTreeCreatorTest();
void TirDependencyTest();
void TirBlockScopeTest();

}