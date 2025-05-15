#include "tvm/ir/global_var_supply.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"

namespace global_var_supply_test {

using tvm::GlobalVar;
using tvm::GlobalVarSupply;
using tvm::NameSupply;

using tvm::BaseFunc;
using tvm::IRModule;
using tvm::RelaxExpr;

using tvm::relax::Call;
using tvm::relax::Function;
using tvm::relax::ShapeStructInfo;
using tvm::relax::Var;

using Expr = tvm::RelaxExpr;

void GlobalVarSupplyTest();

}  // namespace global_var_supply_test
