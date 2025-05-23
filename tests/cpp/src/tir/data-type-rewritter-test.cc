#include "tir/data-type-rewritter-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

namespace data_type_rewriter_test {

/// @brief Legalize the data types of expressions to make sure they are consistent with
/// other parts of the program. It enforces the following rules:
///   - The data type of the index variable in a loop must be consistent with the data
///     type of the loop bounds.
///   - The data type of the binary and ternary expressions must be consistent with the
///     data types of each of their operands.
///   - The data type of the bounds and binding values of block iter vars must be
///     consistent with the data type of the block iter vars.
void TirDataTypeLegalizerTest() {
  LOG_SPLIT_LINE("TirDataTypeLegalizerTest");

  DataTypeLegalizer legalizer;

  /// 1. Select

  auto node = make_object<SelectNode>();
  node->condition = Var("cond", DataType::Bool());
  node->true_value = Var("a", DataType::Int(64));
  node->false_value = IntImm(DataType::Int(32), 2);
  Select oldselect = Downcast<Select>(Select{node});
  LOG_PRINT_VAR(oldselect);  // T.Select(cond, a, 2)
  Select newselect = Downcast<Select>(legalizer(Select(node)));
  LOG_PRINT_VAR(newselect);  // T.Select(cond, a, T.int64(2))

  /// 2. IfThenElse

  auto cond = Var("cond", DataType::Bool());
  PrimExpr call = Call(DataType::Int(32), if_then_else(),
                       {cond, Var("a", DataType::Int(64)), IntImm(DataType::Int(32), 2)});
  LOG_PRINT_VAR(call);  // T.if_then_else(cond, a, 2)
  Call newcall = Downcast<Call>(legalizer(call));
  LOG_PRINT_VAR(newcall);  // T.if_then_else(cond, a, T.int64(2))

  /// 3. Stmt

  auto blocknode = make_object<BlockNode>();
  auto itervarnode = make_object<IterVarNode>();
  itervarnode->var = Var("i", DataType::Int(32));
  itervarnode->dom =
      Range::FromMinExtent(IntImm(DataType::Int(64), 0), IntImm(DataType::Int(64), 10));
  itervarnode->iter_type = IterVarType::kDataPar;
  blocknode->iter_vars = {IterVar(itervarnode)};
  blocknode->reads = {};
  blocknode->writes = {};
  blocknode->name_hint = "block";
  blocknode->body = Evaluate(Integer(0));
  auto blockrealizenode = make_object<BlockRealizeNode>();
  auto loopvar = Var("i", DataType::Int(32));
  blockrealizenode->iter_values = {loopvar};
  blockrealizenode->predicate = const_true();
  blockrealizenode->block = Block(blocknode);
  auto fornode = make_object<ForNode>();
  fornode->loop_var = loopvar;
  fornode->min = IntImm(DataType::Int(32), 0);
  fornode->extent = IntImm(DataType::Int(32), 10);
  fornode->kind = ForKind::kSerial;
  fornode->body = BlockRealize(blockrealizenode);
  Stmt stmt = For(fornode);
  LOG_PRINT_VAR(stmt);
  /// Output:
  ///   for i in range(10):
  ///     with T.block("block"):
  ///         i_1 = T.axis.spatial(T.int64(10), i)
  ///         T.reads()
  ///         T.writes()
  ///         T.evaluate(0)

  DataType targetdtype = loopvar->dtype;
  Stmt newstmt = legalizer(stmt);
  LOG_PRINT_VAR(newstmt);
  /// Output:
  ///   for i in range(10):
  ///     with T.block("block"):
  ///         i_1 = T.axis.spatial(10, i)
  ///         T.reads()
  ///         T.writes()
  ///         T.evaluate(0)

  /// 4. Ramp

  auto node2 = make_object<RampNode>();
  node2->base = IntImm(DataType::Int(64), 0);
  node2->stride = IntImm(DataType::Int(32), 1);
  int lanes = 4;
  node2->lanes = lanes;
  Ramp ramp = Downcast<Ramp>(Ramp(node2));
  LOG_PRINT_VAR(ramp);  // T.Ramp(T.int64(0), 1, 4)
  Ramp newramp = Downcast<Ramp>(legalizer(Ramp(node2)));
  LOG_PRINT_VAR(newramp);  // T.Ramp(T.int64(0), T.int64(1), 4)
}

void TirIndexDataTypeRewriterTest() {
  LOG_SPLIT_LINE("TirIndexDataTypeRewriterTest");

  IndexDataTypeRewriter rewriter;

  auto cond = Var("cond", DataType::Bool());
  PrimExpr call = Call(DataType::Int(32), if_then_else(),
                       {cond, Var("a", DataType::Int(64)), IntImm(DataType::Int(32), 2)});
  LOG_PRINT_VAR(call);  // T.if_then_else(cond, a, 2)
  Call newcall = Downcast<Call>(rewriter(call));
  LOG_PRINT_VAR(newcall);  // T.if_then_else(cond, a, T.int64(2))
}

void TirIndexDataTypeNormalizerTest() {
  LOG_SPLIT_LINE("TirIndexDataTypeNormalizerTest");

  IndexDataTypeNormalizer normalizer{tvm::DataType::Int(64)};

  /// @todo
  auto cond = Var("cond", DataType::Bool());
  PrimExpr call = Call(DataType::Int(32), if_then_else(),
                       {cond, Var("a", DataType::Int(64)), IntImm(DataType::Int(32), 2)});
  LOG_PRINT_VAR(call);  // T.if_then_else(cond, a, 2)
  PrimExpr newcall = normalizer(call);
  LOG_PRINT_VAR(newcall);  // T.if_then_else(cond, a, T.int64(2))
}

}  // namespace data_type_rewriter_test

REGISTER_TEST_SUITE(data_type_rewriter_test::TirDataTypeLegalizerTest,
                    tir_data_type_rewriter_test_TirDataTypeLegalizerTest);
REGISTER_TEST_SUITE(data_type_rewriter_test::TirIndexDataTypeRewriterTest,
                    tir_data_type_rewriter_test_TirIndexDataTypeRewriterTest);
REGISTER_TEST_SUITE(data_type_rewriter_test::TirIndexDataTypeNormalizerTest,
                    tir_data_type_rewriter_test_TirIndexDataTypeNormalizerTest);
