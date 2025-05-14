// clang-format off
#include "tvm/ir/transform.h"
#include "tvm/ir/expr.h"
#include "tvm/ir/function.h"
#include "tvm/ir/module.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include "tvm/relax/transform.h"
#include "tvm/tir/transform.h"
#include "tvm/../../src/node/attr_registry.h"
// clang-format on

namespace transform_test {

using tvm::transform::ApplyPassToFunction;
using tvm::transform::CreateModulePass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassInfo;
using tvm::transform::PrintIR;
using tvm::transform::Sequential;

using tvm::relax::transform::CallTIRRewrite;
using tvm::relax::transform::FoldConstant;
using tvm::tir::transform::LowerTVMBuiltin;

using tvm::BaseFunc;
using tvm::GlobalVar;
using tvm::IRModule;
using tvm::PrimExpr;
using tvm::RelaxExpr;
using Expr = tvm::RelaxExpr;

using tvm::relax::BindingBlock;
using tvm::relax::Call;
using tvm::relax::Constant;
using tvm::relax::FuncStructInfo;
using tvm::relax::Function;
using tvm::relax::SeqExpr;
using tvm::relax::ShapeExpr;
using tvm::relax::ShapeStructInfo;
using tvm::relax::TensorStructInfo;
using tvm::relax::Var;
using tvm::relax::VarBinding;

using tvm::AttrRegistry;
using tvm::Op;
using tvm::OpRegEntry;
using OpRegistry = AttrRegistry<OpRegEntry, Op>;

using tvm::runtime::DataType;
using tvm::runtime::Map;
using tvm::runtime::NDArray;
using tvm::runtime::ShapeTuple;
using tvm::runtime::String;

void PassContextTest();
void PassTest();

}  // namespace transform_test

void PassContextTest();
void PassTest();
