#include "tvm/ir/replace_global_vars.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"

namespace replace_global_vars_test {

using tvm::transform::ReplaceGlobalVars;

using tvm::BaseFunc;
using tvm::DataType;
using tvm::GlobalVar;
using tvm::IRModule;
using tvm::RelaxExpr;

using tvm::relax::Call;
using tvm::relax::Function;
using tvm::relax::ShapeExpr;
using tvm::relax::ShapeStructInfo;
using tvm::relax::TensorStructInfo;
using tvm::relax::Var;

using Expr = tvm::RelaxExpr;

using tvm::runtime::Map;

void ReplaceGlobalVarsTest();

}  // namespace replace_global_vars_test
