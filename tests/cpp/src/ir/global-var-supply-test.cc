#include "ir/global-var-supply-test.h"
#include <tvm/ir/expr.h>

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace global_var_supply_test {

void GlobalVarSupplyTest() {
  LOG_SPLIT_LINE("GlobalVarSupplyTest");

  NameSupply namesupply{"prefix"};
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName_1
  LOG_PRINT_VAR(namesupply->ReserveName("prefix_TestName_1"));
  LOG_PRINT_VAR(namesupply->ContainsName("TestName"));

  /// NameSupply
  GlobalVarSupply gvsupply{namesupply};
  LOG_PRINT_VAR(gvsupply);

  GlobalVar gv1{gvsupply->FreshGlobal("TestNameGV")};
  LOG_PRINT_VAR(gv1);
  GlobalVar gv2{gvsupply->UniqueGlobalFor("TestNameUnique")};
  LOG_PRINT_VAR(gv2);
  GlobalVar gv3{gvsupply->UniqueGlobalFor("TestNameUnique")};
  LOG_PRINT_VAR(gv3);

  GlobalVar gv4{"SingleGV"};

  gvsupply->ReserveGlobalVar(gv4);
  LOG_PRINT_VAR(gv4);

  /// IRModule
  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  /// Create tvm::relax::Function
  Expr opexpr = tvm::Op::Get("relax.nn.conv2d");
  Var arg1{"arg1", tvm::relax::ShapeStructInfo{4}};
  Var arg2{"arg2", tvm::relax::ShapeStructInfo{4}};
  Call call{
      opexpr, {arg1, arg2}
  };
  Function func{
      {arg1, arg2},
      call,
      tvm::relax::ShapeStructInfo{4},
      true,
  };

  /// @note TVMScript cannot print functions of type: BaseFunc
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);
  LOG_SPLIT_LINE("");

  GlobalVarSupply gvsupply2{irmodule2};
  LOG_PRINT_VAR(gvsupply2->FreshGlobal("TestNameGV"));
}

}  // namespace global_var_supply_test

void GlobalVarSupplyTest() { global_var_supply_test::GlobalVarSupplyTest(); }
