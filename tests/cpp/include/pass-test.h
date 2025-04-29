#include "tvm/ir/transform.h"
#include <iomanip>

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace pass_test {

using tvm::runtime::Map;
using tvm::runtime::String;

using tvm::runtime::ObjectRef;

using tvm::transform::PassContext;
using tvm::transform::PassContextNode;

template <typename K, typename V>
std::ostream &operator<<(std::ostream &os, Map<K, V> &map);

std::ostream &operator<<(std::ostream &os, const PassContext &ctx);

}  // namespace pass_test

void PassTest();
