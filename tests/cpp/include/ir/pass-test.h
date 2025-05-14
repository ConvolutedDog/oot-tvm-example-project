#include "tvm/ir/transform.h"
#include <iomanip>

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

void PassTestTemp();
