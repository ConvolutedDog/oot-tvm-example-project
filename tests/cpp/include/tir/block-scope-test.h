#include "tvm/tir/block_scope.h"

namespace block_scope_test {

using tvm::tir::BlockScope;
using tvm::tir::Dependency;
using tvm::tir::DepKind;
using tvm::tir::SRefTreeCreator;
using tvm::tir::StmtSRef;

using tvm::IRModule;
using tvm::runtime::Array;
using tvm::tir::BlockRealizeNode;
using tvm::tir::ForNode;
using tvm::tir::SeqStmtNode;
using tvm::tir::StmtNode;

void TirStmtSRefTest();
void TirSRefTreeCreatorTest();
void TirDependencyTest();
void TirBlockScopeTest();

}  // namespace block_scope_test
