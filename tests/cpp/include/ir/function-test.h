#include "tvm/ir/expr.h"
#include "tvm/ir/function.h"

namespace function_test {

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

void BaseFuncTest();

}  // namespace function_test

void BaseFuncTest();
