#include "dlpack/dlpack.h"
#include "tvm/relax/expr.h"
#include "tvm/runtime/ndarray.h"

namespace expr_test {

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

using tvm::DataType;

using tvm::runtime::NDArray;
using tvm::runtime::ShapeTuple;

void CallTest();
void TupleTest();
void TupleGetItemTest();
void LeafExprTest();
void BindTest();

}  // namespace expr_test

void CallTest();
void TupleTest();
void TupleGetItemTest();
void LeafExprTest();
void BindTest();
