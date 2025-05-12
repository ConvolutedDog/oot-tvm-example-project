#include "tvm/ir/expr.h"
#include "tvm/ir/function.h"
#include "tvm/ir/module.h"

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

void ModuleTest();

}  // namespace module_test

void ModuleTest();
