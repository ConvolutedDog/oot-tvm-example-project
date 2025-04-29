#include "dlpack/dlpack.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/data_type.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace expr_test {}  // namespace expr_test

void PrimExprTest();
void BoolTest();
void IntegerTest();
void RangeTest();
