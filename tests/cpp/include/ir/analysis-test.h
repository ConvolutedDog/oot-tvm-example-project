#include "tvm/ir/analysis.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"

namespace analysis_test {

using tvm::ir::CalleeCollector;
using tvm::ir::CollectCallMap;

using tvm::BaseFunc;
using tvm::GlobalVar;
using tvm::IRModule;
using tvm::RelaxExpr;

using tvm::relax::Call;
using tvm::relax::Function;
using tvm::relax::Var;

using Expr = tvm::RelaxExpr;

void AnalysisTest();

}  // namespace analysis_test
