#include "tvm/tir/data_layout.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

namespace data_layout_test {

using tvm::tir::BijectiveLayout;
using tvm::tir::Layout;
using tvm::tir::LayoutAxis;

using tvm::PrimExpr;
using tvm::Range;
using tvm::runtime::Array;
using tvm::runtime::DataType;
using tvm::runtime::String;
using tvm::tir::IterVar;
using tvm::tir::IterVarType;
using tvm::tir::Var;

void TirLayoutAxisTest();
void TirLayoutTest();
void TirBijectiveLayoutTest();

}  // namespace data_layout_test
