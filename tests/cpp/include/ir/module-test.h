#include "tvm/ir/expr.h"
#include "tvm/ir/function.h"
#include "tvm/ir/module.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"

namespace module_test {

using tvm::runtime::make_object;
using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::ObjectRef;

using tvm::BaseFunc;
using tvm::BaseFuncNode;

using tvm::RelaxExpr;
using tvm::RelaxExprNode;

using tvm::runtime::Map;
using tvm::runtime::String;

using tvm::DictAttrs;

using tvm::LinkageType;
using tvm::attr::kGlobalSymbol;

using tvm::NullValue;

using tvm::IRModule;
using tvm::IRModuleNode;

using tvm::GlobalVar;

using tvm::relax::Function;
using Expr = tvm::RelaxExpr;
using tvm::relax::Call;
using tvm::relax::FuncStructInfo;
using tvm::relax::ShapeStructInfo;
using tvm::relax::TensorStructInfo;
using tvm::relax::Var;

void IrModuleTest();

}  // namespace module_test
