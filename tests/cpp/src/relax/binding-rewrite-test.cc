#include "utils.h"
#include <relax/binding-rewrite-test.h>
#include <tvm/ir/module.h>
#include <tvm/relax/binding_rewrite.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/logging.h>

namespace binding_rewrite_test {

void DataflowBlockRewriteTest() {
  TensorStructInfo info(ShapeExpr({32, 32}), DataType::Float(32));

  GlobalVar global_var("main");
  Var x("x", info);

  Var lv0("lv0", info);

  VarBinding bind1(lv0, x);
  DataflowBlock dfb = DataflowBlock({bind1});

  // BindingBlock bindBlock1 = dfb;

  SeqExpr body({dfb}, lv0);
  Function fn({x}, body, info);
  LOG_PRINT_VAR(fn);

  IRModule mod({
      {global_var, fn}
  });
  LOG_PRINT_VAR(mod);

  auto rwt = DataflowBlockRewrite(dfb, fn);
  LOG_PRINT_VAR(rwt);

  ///@note If the variable name is not given, it will be automatically generated in a form
  /// of "tmp${COUNTER}". The variable type will be DataflowVar if is_dfvar is True,
  /// otherwise it will be Var. Being Var means the variables are output variables of the
  /// DataflowBlock. While being DataflowVar means the variables are internal variables of
  /// the DataflowBlock.
  rwt->Add("tmp1", lv0, false);
  rwt->Add("tmp2", x, true);

  LOG_PRINT_VAR(rwt->MutatedFunc());
  LOG_PRINT_VAR(rwt->MutatedDataflowBlock());

  DataflowVar tmp3("tmp3", info);
  VarBinding bind2(tmp3, lv0);
  rwt->Add(bind2);
  LOG_PRINT_VAR(rwt->MutatedFunc());

  IRModule newMod = rwt->MutateIRModule(mod);
  LOG_PRINT_VAR(newMod);

  ///@attention can't run `rwt->RemoveUnused(tmp3)` as `tmp3` isn't in the UD-chain of
  ///`rwt`.

  /// @brief [TEST] remove the unused variable.
  DataflowVar unused("unused", info);
  VarBinding bind3(unused, x);

  DataflowBlock dfb2({bind1, bind2, bind3});
  SeqExpr body2({dfb2}, lv0);
  Function fn2({x}, body2, info);

  auto rwt2 = DataflowBlockRewrite(dfb2, fn2);

  LOG_PRINT_VAR(rwt2->MutatedFunc());
  LOG_PRINT_VAR(rwt2->MutatedDataflowBlock());
  rwt2->RemoveUnused(unused);
  LOG_PRINT_VAR(rwt2->MutatedFunc());

  rwt2->RemoveAllUnused();
  LOG_PRINT_VAR(rwt2->MutatedFunc());

  /// @brief [TEST] replace all uses.
  {
    /// @note [The difference between `Var` and `DataflowVar`]
    /// Being Var means the variables are output variables of the
    /// DataflowBlock. While being DataflowVar means the variables are internal variables
    /// of the DataflowBlock.
    DataflowVar lv0("lv0", info);
    DataflowVar lv1("lv1", info);
    DataflowVar lv2("lv2", info);
    DataflowVar lv3("lv3", info);

    Var lv4("lv4", info);

    /* UD-chain
       lv0     lv1
      /  \
    lv2  lv3
     \    /
      lv4
    */

    VarBinding bind1(lv0, x);
    VarBinding bind2(lv1,Call(Op::Get("relax.multiply"),{x,x}));
    VarBinding bind3(lv2, lv0);
    VarBinding bind4(lv3, lv0);
    VarBinding bind5(lv4,Call(Op::Get("relax.add"), {lv2, lv3}));


    DataflowBlock dfb({bind1, bind2, bind3, bind4,bind5});

    SeqExpr body({dfb}, lv4);
    Function fn({x}, body, info);

    auto rwt = DataflowBlockRewrite(dfb, fn);
    LOG_PRINT_VAR(rwt->MutatedFunc());

    /// replace all uses of lv0 with lv1.
    rwt->ReplaceAllUses(lv0, lv1);
    LOG_PRINT_VAR(rwt->MutatedFunc());

    /// The `lv0` is unused now, and we can remove it.
    rwt->RemoveUnused(lv0);
    LOG_PRINT_VAR(rwt->MutatedFunc());
  }
}
}  // namespace binding_rewrite_test

REGISTER_TEST_SUITE(binding_rewrite_test::DataflowBlockRewriteTest,
                    relax_binding_rewrite_test_DataflowBlockRewriteTest);
