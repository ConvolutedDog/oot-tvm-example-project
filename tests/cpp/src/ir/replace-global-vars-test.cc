#include "ir/replace-global-vars-test.h"
#include "test-func-registry.h"

namespace replace_global_vars_test {

void ReplaceGlobalVarsTest() {
  LOG_SPLIT_LINE("ReplaceGlobalVarsTest");

  /// IRModule
  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  // Define input tensors (with TensorStructInfo)
  Var arg1{"arg1", TensorStructInfo(ShapeExpr({1, 3, 224, 224}), DataType::Float(32))};
  Var arg2{"arg2", TensorStructInfo(ShapeExpr({64, 3, 3, 3}), DataType::Float(32))};

  // Get the "relax.nn.conv2d" operator
  Expr opexpr = tvm::Op::Get("relax.nn.conv2d");

  // Create a Call node (conv2d)
  Call call{
      opexpr, {arg1, arg2}
  };

  // Create a Relax Function
  Function func{
      {arg1, arg2},
      call,
      TensorStructInfo(ShapeExpr({1, 64, 222, 222}
      ),
                       DataType::Float(32)), // Output tensor info
      true, // is_pure
  };

  // Build IRModule
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);
  LOG_SPLIT_LINE("");

  /// ReplaceGlobalVars
  ///
  /// @bug
  ///
  /// GlobalVar globalvar2("globalvar2");
  /// Map<GlobalVar, GlobalVar> gvarreplacements;
  /// gvarreplacements.Set(globalvar, globalvar2);
  /// IRModule irmodule3 = ReplaceGlobalVars(irmodule2, gvarreplacements);
  /// LOG_PRINT_VAR(irmodule3);
  /// LOG_SPLIT_LINE("");
}

}  // namespace replace_global_vars_test

REGISTER_TEST_SUITE(replace_global_vars_test::ReplaceGlobalVarsTest,
                    ir_replace_global_vars_test_ReplaceGlobalVarsTest);
