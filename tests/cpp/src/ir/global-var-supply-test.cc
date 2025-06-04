#include "ir/global-var-supply-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

namespace global_var_supply_test {

/// @brief GlobalVarSupply can be used to generate unique GlobalVars.
void IrGlobalVarSupplyTest() {
  LOG_SPLIT_LINE("IrGlobalVarSupplyTest");

  NameSupply namesupply{"prefix"};
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName_1
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName_2
  LOG_PRINT_VAR(
      namesupply->ReserveName("prefix_TestName_1"));           // prefix_prefix_TestName_1
  LOG_PRINT_VAR(namesupply->ContainsName("TestName", false));  // 0
  LOG_PRINT_VAR(namesupply->ContainsName("TestName", true));   // 1
  LOG_PRINT_VAR(namesupply->ContainsName("prefix_TestName_1", false));         // 1
  LOG_PRINT_VAR(namesupply->ContainsName("prefix_TestName_2", false));         // 1
  LOG_PRINT_VAR(namesupply->ContainsName("prefix_prefix_TestName_1", false));  // 1

  /// NameSupply
  GlobalVarSupply gvsupply{namesupply};
  LOG_PRINT_VAR(gvsupply);  // gvsupply: GlobalVarSupply(0xfac0c0)

  GlobalVar gv1{gvsupply->FreshGlobal("TestNameGV")};
  LOG_PRINT_VAR(gv1);  // gv1: I.GlobalVar("prefix_TestNameGV")
  GlobalVar gv2{gvsupply->UniqueGlobalFor("TestNameUnique")};
  LOG_PRINT_VAR(gv2);  // gv2: I.GlobalVar("prefix_TestNameUnique")
  GlobalVar gv3{gvsupply->UniqueGlobalFor("TestNameUnique")};
  LOG_PRINT_VAR(gv3);  // gv3: I.GlobalVar("prefix_TestNameUnique")

  GlobalVar gv4{"SingleGV"};
  gvsupply->ReserveGlobalVar(gv4);
  LOG_PRINT_VAR(gv4);  // gv4: I.GlobalVar("SingleGV")

  /// IRModule
  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  /// Create tvm::relax::Function
  Expr opexpr = tvm::Op::Get("relax.add");
  Var arg1{
      "arg1", tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4}
  };
  Var arg2{
      "arg2", tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4}
  };
  Call call{
      opexpr, {arg1, arg2}
  };
  Function func{
      {arg1,                     arg2},
      call,
      tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4   },
      true,
  };

  /// @note TVMScript cannot print functions of type: BaseFunc
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);
  /// Output:
  ///   # from tvm.script import ir as I
  ///   # from tvm.script import relax as R
  ///
  ///   @I.ir_module
  ///   class Module:
  ///       @R.function(private=True)
  ///       def globalvar(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  LOG_SPLIT_LINE("");

  GlobalVarSupply gvsupply2{irmodule2};
  LOG_PRINT_VAR(gvsupply2->FreshGlobal("TestNameGV"));
  /// gvsupply2->FreshGlobal("TestNameGV"): I.GlobalVar("tvmgen_default_TestNameGV")
}

}  // namespace global_var_supply_test

void IrGlobalVarSupplyTest() { global_var_supply_test::IrGlobalVarSupplyTest(); }

REGISTER_TEST_SUITE(global_var_supply_test::IrGlobalVarSupplyTest,
                    ir_global_var_supply_test_IrGlobalVarSupplyTest);
