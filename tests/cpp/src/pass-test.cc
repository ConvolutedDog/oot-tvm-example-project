#include "../include/pass-test.h"
#include "tvm/runtime/container/map.h"
#include "tvm/runtime/container/string.h"

using tvm::runtime::Map;
using tvm::runtime::String;

using tvm::runtime::ObjectRef;

using tvm::transform::PassContext;
using tvm::transform::PassContextNode;

namespace pass_test {

template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
std::ostream &operator<<(std::ostream &os, Map<K, V> &map) {
  LOG_SPLIT_LINE(typeid(map).name());
  for (typename Map<K, V>::iterator iter = map.begin(); iter != map.end(); ++iter) {
    auto &pair = *(iter);
    os << std::setw(50 - pair.first.size()) << std::right << pair.first << " | "
       << std::left << pair.second << '\n';
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PassContext &ctx) {
  LOG_SPLIT_LINE(typeid(ctx).name());
  const PassContextNode *passCtxNode = ctx.operator->();
  LOG_PRINT_VAR(passCtxNode->opt_level);
  LOG_PRINT_VAR(passCtxNode->required_pass);
  LOG_PRINT_VAR(passCtxNode->disabled_pass);
  LOG_PRINT_VAR(passCtxNode->diag_ctx);
  LOG_PRINT_VAR(passCtxNode->config);
  LOG_PRINT_VAR(passCtxNode->instruments);
  LOG_PRINT_VAR(passCtxNode->trace_stack);
  LOG_PRINT_VAR(passCtxNode->make_traceable);
  LOG_PRINT_VAR(passCtxNode->num_evals);
  LOG_PRINT_VAR(passCtxNode->tuning_api_database);
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

  pass_test::operator<<(std::cout, passctx);
  pass_test::operator<<(std::cout, PassContext::Current());
}
