// clang-format off
#include "dlpack/dlpack.h"
#include "tvm/ir/op.h"
#include "tvm/node/script_printer.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/tir/var.h"
#include "tvm/../../src/node/attr_registry.h"
// clang-format on

namespace expr_test {

using tvm::PrimExpr;
using tvm::RelaxExpr;
using Expr = tvm::RelaxExpr;

using tvm::GlobalVar;
using tvm::relax::Call;
using tvm::relax::Function;
using tvm::relax::Var;

using tvm::relax::Tuple;
using tvm::relax::TupleGetItem;
using tvm::relax::WithFields;

using tvm::relax::Constant;
using tvm::relax::DataflowVar;
using tvm::relax::DataTypeImm;
using tvm::relax::PrimValue;
using tvm::relax::ShapeExpr;
using tvm::relax::StringImm;

using tvm::relax::Binding;
using tvm::relax::BindingBlock;
using tvm::relax::ExternFunc;
using tvm::relax::GetShapeOf;
using tvm::relax::If;
using tvm::relax::MatchCast;
using tvm::relax::SeqExpr;
using tvm::relax::VarBinding;

using tvm::AttrRegistry;
using tvm::Op;
using tvm::OpRegEntry;
using OpRegistry = AttrRegistry<OpRegEntry, Op>;

using tvm::DataType;

using tvm::runtime::NDArray;
using tvm::runtime::ShapeTuple;

using tvm::relax::FuncStructInfo;
using tvm::relax::ShapeStructInfo;
using tvm::relax::TensorStructInfo;

void CallTest();
void TupleTest();
void TupleGetItemTest();
void LeafExprTest();
void BindTest();

}  // namespace expr_test
