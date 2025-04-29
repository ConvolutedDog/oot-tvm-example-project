#include "../include/expr-test.h"

using tvm::PrimExpr;
using tvm::runtime::DataType;
using tvm::runtime::operator<<;

namespace expr_test {}

void PrimExprTest() {
  PrimExpr primexpr = 3;
  DataType dtype = primexpr.dtype();
  DLDataType dldtype = DLDataType(dtype);
  LOG_PRINT_VAR(dldtype);
}
