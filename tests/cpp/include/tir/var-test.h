#include "tvm/ir/expr.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/var.h"

namespace var_test {

using tvm::tir::IterVar;
using tvm::tir::IterVarNode;
using tvm::tir::IterVarType;
using tvm::tir::IterVarType2String;
using tvm::tir::SizeVar;
using tvm::tir::SizeVarNode;
using tvm::tir::Var;
using tvm::tir::VarNode;

using tvm::DataType;
using tvm::PointerType;
using tvm::Range;
using tvm::VoidType;

using tvm::runtime::String;

using tvm::BaseExpr;
using tvm::BaseExprNode;
using tvm::IntImm;
using tvm::IntImmNode;
using tvm::PrimExpr;
using tvm::PrimExprNode;

}  // namespace var_test

void VarTest();
void SizeVarTest();
void IterVarTest();
