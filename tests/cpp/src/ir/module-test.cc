#include "ir/module-test.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace module_test {

void ModuleTest() {
  LOG_SPLIT_LINE("ModuleTest");

  /// @brief Define a BaseFunc.
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

  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  IRModule irmodule = IRModule::FromExpr(basefunc);
  /// @note TVMScript cannot print functions of type: BaseFunc
  /// IRModule irmodule{{std::pair<GlobalVar, BaseFunc>{globalvar, basefunc}}};

  LOG_PRINT_VAR(irmodule);
}

}  // namespace module_test

void ModuleTest() { module_test::ModuleTest(); }
