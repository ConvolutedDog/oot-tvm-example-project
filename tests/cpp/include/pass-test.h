#include "tvm/ir/transform.h"
#include <iomanip>

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                   \
  std::cout << "==============" << (stmt) << "==============\n";

namespace pass_test {}

void PassTest();
