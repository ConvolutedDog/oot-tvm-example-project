#include "tvm/node/serialization.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"

namespace serialization_test {

using tvm::LoadJSON;
using tvm::SaveJSON;

using tvm::BaseFunc;
using tvm::GlobalVar;
using tvm::IRModule;
using tvm::relax::Call;
using tvm::relax::Function;
using tvm::relax::Var;
using Expr = tvm::RelaxExpr;

void NodeSerializationTest();

}  // namespace serialization_test
