#include "../include/pass-test.h"
#include "tvm/runtime/container/map.h"
#include "tvm/runtime/container/string.h"

using tvm::runtime::Map;
using tvm::runtime::String;

using tvm::runtime::ObjectRef;

using tvm::transform::PassContext;

namespace pass_test {

template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
std::ostream &operator<<(std::ostream &os, Map<K, V> &map) {
  for (typename Map<K, V>::iterator iter = map.begin(); iter != map.end(); ++iter) {
    auto &pair = *(iter);
    os << std::setw(50 - pair.first.size()) << std::right << pair.first << " | "
       << std::left << pair.second << '\n';
  }
  return os;
}

}  // namespace pass_test

void PassTest() {
  PassContext passctx;
  passctx = PassContext::Create();
  Map<String, Map<String, String>> listconfigs = PassContext::ListConfigs();
  LOG_PRINT_VAR(listconfigs.size());

  std::cout << listconfigs << "\n";

  for (Map<String, Map<String, String>>::iterator iter = listconfigs.begin();
       iter != listconfigs.end(); ++iter) {
    LOG_PRINT_VAR((*iter).first);
    LOG_PRINT_VAR((*iter).second);
  }

  pass_test::operator<<(std::cout, listconfigs);
}
