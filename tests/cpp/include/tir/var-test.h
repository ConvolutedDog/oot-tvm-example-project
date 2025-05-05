#include "tvm/tir/var.h"
#include "tvm/runtime/data_type.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/container/string.h"

namespace var_test {

using tvm::tir::VarNode;
using tvm::tir::Var;
using tvm::tir::SizeVarNode;
using tvm::tir::SizeVar;
using tvm::tir::IterVarNode;
using tvm::tir::IterVar;
using tvm::tir::IterVarType;
using tvm::tir::IterVarType2String;

using tvm::DataType;
using tvm::PointerType;
using tvm::VoidType;
using tvm::Range;

using tvm::runtime::String;

using tvm::BaseExprNode;
using tvm::BaseExpr;
using tvm::PrimExprNode;
using tvm::PrimExpr;
using tvm::IntImm;
using tvm::IntImmNode;

void VarTest();
void SizeVarTest();
void IterVarTest();

}

void VarTest();
void SizeVarTest();
void IterVarTest();
