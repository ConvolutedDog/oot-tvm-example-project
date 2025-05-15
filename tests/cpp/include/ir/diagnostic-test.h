#include "tvm/ir/diagnostic.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"

namespace diagnostic_test {

using tvm::Diagnostic;
using tvm::DiagnosticBuilder;
using tvm::DiagnosticContext;
using tvm::DiagnosticLevel;
using tvm::DiagnosticRenderer;
using tvm::runtime::TypedPackedFunc;

using tvm::BaseFunc;
using tvm::GlobalVar;
using tvm::IRModule;
using tvm::RelaxExpr;

using tvm::relax::Call;
using tvm::relax::Function;
using tvm::relax::Var;

using tvm::relax::ShapeStructInfo;

using Expr = tvm::RelaxExpr;

using tvm::SourceName;
using tvm::Span;

}  // namespace diagnostic_test

void DiagnosticTest();
void DiagnosticContextTest();
