#include "ir/function-test.h"
#include "test-func-registry.h"

namespace function_test {

std::ostream &operator<<(std::ostream &os, const LinkageType &linkage) {
  std::string ret;
  switch (linkage) {
    case LinkageType::kInternal: ret = "LinkageType::kInternal"; break;
    case LinkageType::kExternal: ret = "LinkageType::kExternal"; break;
    default: ret = "Unknown LinkageType";
  }
  os << ret;
  return os;
}

void BaseFuncTest() {
  LOG_SPLIT_LINE("BaseFuncTest");
  ObjectPtr<BaseFuncNode> basefuncnodeptr = make_object<BaseFuncNode>();
  std::initializer_list<std::pair<String, ObjectRef>> init = {
      {"attr1",       String("attr1")       },
      {"attr2",       tvm::PrimExpr(0)      },
      {"attr3",       String("attr3")       },
      /// Setting linkage type follows these rules:​​
      /// 1. ​​LinkageType::kExternal is set ONLY when:​​
      ///   - The kGlobalSymbol key is explicitly set ​​AND​​
      ///   - Its value is ​​not​​ NullValue<ObjectRef>().
      /// 2. ​​LinkageType::kInternal is set in ALL OTHER CASES (default):​​
      ///   - When kGlobalSymbol is ​​not set​​, ​​OR​​
      ///   - When kGlobalSymbol is set but its value ​​is​​
      ///     `NullValue<ObjectRef>()`.
      {kGlobalSymbol, NullValue<ObjectRef>()}
  };
  Map<String, ObjectRef> map{init};
  DictAttrs dict(map);
  basefuncnodeptr->attrs = dict;

  BaseFunc basefunc(basefuncnodeptr);

  auto *basefuncnode = basefunc.as<BaseFuncNode>();

  LOG_PRINT_VAR(basefuncnode->attrs);

  LOG_PRINT_VAR(basefuncnode->GetAttr<String>("attr1"));
  LOG_PRINT_VAR(basefuncnode->GetAttr<tvm::PrimExpr>("attr2"));
  LOG_PRINT_VAR(basefuncnode->GetAttr<String>("attr3"));

  LOG_PRINT_VAR(basefuncnode->HasNonzeroAttr("attr2"));

  LOG_PRINT_VAR(basefuncnode->GetLinkageType());
}

}  // namespace function_test

REGISTER_TEST_SUITE(function_test::BaseFuncTest, ir_function_test_BaseFuncTest);
