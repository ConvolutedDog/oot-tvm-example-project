#include "ir/module-test.h"
#include "test-func-registry.h"
#include "utils.h"
#include <tvm/ir/module.h>
#include <tvm/runtime/data_type.h>
#include "tvm/ir/replace_global_vars.h"

namespace module_test {

void IrModuleTest() {
  LOG_SPLIT_LINE("IrModuleTest");

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
  IRModule irmodule1 = IRModule::FromExpr(basefunc);
  LOG_PRINT_VAR(irmodule1);
  /// Output:
  ///   irmodule1: # from tvm.script import ir as I
  ///   @I.ir_module
  ///   class Module:
  ///       main = metadata["BaseFunc"][0]
  LOG_SPLIT_LINE("");

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
  LOG_SPLIT_LINE("");
  /// Output:
  /// irmodule2: # from tvm.script import ir as I
  /// # from tvm.script import relax as R
  ///
  /// @I.ir_module
  /// class Module:
  ///     @R.function(private=True)
  ///     def globalvar(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                   arg2: R.Tensor(dtype="float32", ndim=4)
  ///         ) -> R.Tensor(dtype="float32", ndim=4):
  ///         return R.add(arg1, arg2)

  /// @brief
  IRModule irmodule3 = IRModule::FromExpr(func);
  LOG_PRINT_VAR(irmodule3);
  LOG_SPLIT_LINE("");
  /// Output:
  /// irmodule2: # from tvm.script import ir as I
  /// # from tvm.script import relax as R
  ///
  /// @I.ir_module
  /// class Module:
  ///     @R.function(private=True)
  ///     def main(arg1: R.Tensor(dtype="float32", ndim=4),
  ///              arg2: R.Tensor(dtype="float32", ndim=4)
  ///         ) -> R.Tensor(dtype="float32", ndim=4):
  ///         return R.add(arg1, arg2)

  LOG_PRINT_VAR(irmodule2->functions);
  /// Output:
  ///   {I.GlobalVar("globalvar"): # from tvm.script import relax as R ...}
  LOG_PRINT_VAR(irmodule2->source_map);  // SourceMap(0x28ecba0)
  LOG_PRINT_VAR(irmodule2->attrs);
  LOG_PRINT_VAR(irmodule2->global_infos);
  LOG_PRINT_VAR(irmodule2->global_var_map_);  // {"globalvar": I.GlobalVar("globalvar")}

  /// Add a function
  GlobalVar globalvar2("globalvar2");
  irmodule2->Add(globalvar2, func);
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
  ///
  ///       @R.function(private=True)
  ///       def globalvar2(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                      arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  LOG_SPLIT_LINE("");

  /// Add a function
  GlobalVar globalvar3("globalvar3");
  irmodule2->AddUnchecked(globalvar3, func);
  LOG_PRINT_VAR(irmodule2);
  /// Output:
  ///   @I.ir_module
  ///   class Module:
  ///       @R.function(private=True)
  ///       def globalvar(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       @R.function(private=True)
  ///       def globalvar2(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       @R.function(private=True)
  ///       def globalvar3(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  LOG_SPLIT_LINE("");

  /// Update a function
  Expr opexpr2 = tvm::Op::Get("relax.max");
  Call call2{
      opexpr2, {arg1, arg2}
  };
  Function func2{
      {arg1,                     arg2},
      call2,
      tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4   },
      true,
  };
  irmodule2->Update(globalvar3, func2);
  LOG_PRINT_VAR(irmodule2);
  /// Output:
  ///   @I.ir_module
  ///   class Module:
  ///       @R.function(private=True)
  ///       def globalvar(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       @R.function(private=True)
  ///       def globalvar2(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       @R.function(private=True)
  ///       def globalvar3(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.max(arg1, arg2)
  LOG_SPLIT_LINE("");

  /// Remove a function
  irmodule2->Remove(globalvar3);
  LOG_PRINT_VAR(irmodule2);
  /// Output:
  ///   @I.ir_module
  ///   class Module:
  ///       @R.function(private=True)
  ///       def globalvar(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       @R.function(private=True)
  ///       def globalvar2(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  LOG_SPLIT_LINE("");

  LOG_PRINT_VAR(irmodule2->ContainGlobalVar("globalvar"));   // 1
  LOG_PRINT_VAR(irmodule2->ContainGlobalVar("globalvar2"));  // 1
  LOG_PRINT_VAR(irmodule2->ContainGlobalVar("globalvar3"));  // 0

  LOG_PRINT_VAR(irmodule2->GetGlobalVar("globalvar"));  // I.GlobalVar("globalvar")
  LOG_PRINT_VAR(irmodule2->GetGlobalVars());
  /// Output:
  ///   [I.GlobalVar("globalvar"), I.GlobalVar("globalvar2")]

  LOG_PRINT_VAR(irmodule2->GetAttrs());  // {}

  LOG_PRINT_VAR(irmodule2->Lookup(globalvar));
  /// Output:
  ///   # from tvm.script import relax as R
  ///   @R.function(private=True)
  ///   def main(arg1: R.Tensor(dtype="float32", ndim=4),
  ///            arg2: R.Tensor(dtype="float32", ndim=4)
  ///       ) -> R.Tensor(dtype="float32", ndim=4):
  ///       return R.add(arg1, arg2)
  LOG_PRINT_VAR(irmodule2->Lookup("globalvar"));
  /// Output:
  ///   # from tvm.script import relax as R
  ///   @R.function(private=True)
  ///   def main(arg1: R.Tensor(dtype="float32", ndim=4),
  ///            arg2: R.Tensor(dtype="float32", ndim=4)
  ///       ) -> R.Tensor(dtype="float32", ndim=4):
  ///       return R.add(arg1, arg2)

  /// Update a IRModule
  LOG_SPLIT_LINE("irmodule1 before update:");
  LOG_PRINT_VAR(irmodule1);
  /// Output:
  ///   irmodule1: # from tvm.script import ir as I
  ///   @I.ir_module
  ///   class Module:
  ///       main = metadata["BaseFunc"][0]

  irmodule1->Update(irmodule2);
  LOG_SPLIT_LINE("irmodule1 after update:");
  LOG_PRINT_VAR(irmodule1);
  /// Output:
  ///   irmodule1: # from tvm.script import ir as I
  ///   # from tvm.script import relax as R
  ///
  ///   @I.ir_module
  ///   class Module:
  ///       @R.function(private=True)
  ///       def globalvar(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                     arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       @R.function(private=True)
  ///       def globalvar2(arg1: R.Tensor(dtype="float32", ndim=4),
  ///                      arg2: R.Tensor(dtype="float32", ndim=4)
  ///           ) -> R.Tensor(dtype="float32", ndim=4):
  ///           return R.add(arg1, arg2)
  ///
  ///       main = metadata["BaseFunc"][0]

  /// Shallow copy
  irmodule3 = irmodule1->ShallowCopy();
  LOG_PRINT_VAR(irmodule3);  // same to irmodule1

  /// Test `_contains_relax`
  LOG_PRINT_VAR(_contains_relax(irmodule3));  // _contains_relax(irmodule3): 1
}

}  // namespace module_test

REGISTER_TEST_SUITE(module_test::IrModuleTest, ir_module_test_IrModuleTest);
