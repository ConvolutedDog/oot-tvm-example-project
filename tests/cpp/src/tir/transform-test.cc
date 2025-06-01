#include "tir/transform-test.h"
#include "test-func-registry.h"
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/container/map.h>
#include <tvm/support/with.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

namespace transform_test {

IRModule SimpleFor() {
  // Itervar and buffer
  Var i("i", DataType::Int(32));
  Buffer buffA = decl_buffer({10}, DataType::Int(32), "A", "", {});
  Buffer buffB = decl_buffer({10}, DataType::Int(32), "B", "", {});

  // Buffer load and store
  PrimExpr loadA = tvm::tir::BufferLoad{buffA, {i}};
  Stmt storeB = tvm::tir::BufferStore{buffB, loadA + i, {i}};
  Stmt storeA = tvm::tir::BufferStore{buffA, i, {i}};

  // Loop
  Stmt loop = For(i, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 10),
                  ForKind::kVectorized,
                  SeqStmt{
                      {storeB, storeA}
  });

  // Build PrimFunc and IRModule
  Map<Var, Buffer> buffermap;
  buffermap.Set(buffA->data, buffA);
  buffermap.Set(buffB->data, buffB);
  PrimFunc func(/*params=*/{}, loop, VoidType(), buffermap);

  IRModule mod = IRModule::FromExpr(func);
  return mod;
}

void TirVectorizeLoopTest() {
  LOG_SPLIT_LINE("TirVectorizeLoopTest");

  /// A simple case.
  auto mod = SimpleFor();

  LOG_PRINT_VAR(mod);
  /// Output:
  ///   # from tvm.script import ir as I
  ///   # from tvm.script import tir as T
  ///   @I.ir_module
  ///   class Module:
  ///       @T.prim_func(private=True)
  ///       def main():
  ///           B = T.Buffer((10,), "int32")
  ///           A = T.Buffer((10,), "int32")
  ///           for i in T.vectorized(10):
  ///               B[i] = A[i] + i
  ///               A[i] = i

  LOG_PRINT_VAR(VectorizeLoop()(mod));
  /// Output:
  ///   # from tvm.script import ir as I
  ///   # from tvm.script import tir as T
  ///   @I.ir_module
  ///   class Module:
  ///       @T.prim_func(private=True)
  ///       def main():
  ///           B = T.Buffer((10,), "int32")
  ///           A = T.Buffer((10,), "int32")
  ///           B[0:10] = A[0:10] + T.Ramp(0, 1, 10)
  ///           A[0:10] = T.Ramp(0, 1, 10)

  /// Try to manually transform the initial IRModule to the above IRModule.
  /// @ref tvm/src/tir/transforms/vectorize_loop.cc

  LOG_SPLIT_LINE("Try to manually transform the initial IRModule to the above IRModule.");
  LOG_PRINT_VAR(mod->functions);
  /// Output:
  ///   {
  ///    I.GlobalVar("main"):
  ///    # from tvm.script import tir as T
  ///    @T.prim_func(private=True)
  ///    def main():
  ///        B = T.Buffer((10,), "int32")
  ///        A = T.Buffer((10,), "int32")
  ///        for i in T.vectorized(10):
  ///            B[i] = A[i] + i
  ///            A[i] = i
  ///   }
  LOG_PRINT_VAR(mod->functions.at(mod->global_var_map_["main"]));
  auto func = mod->functions.at(mod->global_var_map_["main"]);
  PrimFunc primfunc = func.as<PrimFunc>().value();
  LOG_PRINT_VAR(primfunc->attrs);  // {}
  LOG_PRINT_VAR(primfunc->body);
  /// Output:
  ///   for i in T.vectorized(10):
  ///     B = T.Buffer((10,), "int32")
  ///     A = T.Buffer((10,), "int32")
  ///     B[i] = A[i] + i
  ///     A[i] = i
  const tvm::tir::ForNode *op = primfunc->body.as<tvm::tir::ForNode>();
  assert(op->kind == tvm::tir::ForKind::kVectorized);
  auto *extent_as_int = op->extent.as<tvm::IntImmNode>();  // NOLINT
  LOG_PRINT_VAR(extent_as_int->value);                     // 10
  LOG_PRINT_VAR(extent_as_int->dtype);                     // int32
  LOG_PRINT_VAR(op->min);                                  // 0
  /// @todo `class Vectorizer` is in tvm/src/tir/transforms/vectorize_loop.cc and cannot
  /// be included.
}

IRModule PartitionedLoop() {

  /// @brief the sample below refers to
  Var i("i", DataType::Int(32));
  Var j("j", DataType::Int(32));
  Buffer buffA = decl_buffer({40}, DataType::Int(32), "A");
  Buffer buffB = decl_buffer({40}, DataType::Int(32), "B");

  Stmt loop_j =
      For(j, 0, 10, ForKind::kSerial,
          IfThenElse((i * 10 + j < 36),
                     BufferStore(buffA, BufferLoad(buffB, {10 * i + j}), {10 * i + j})));

  Stmt loop_i = For(i, 0, 4, ForKind::kSerial, loop_j);

  Map<Var, Buffer> buffer_map;
  buffer_map.Set(buffA->data, buffA);
  buffer_map.Set(buffB->data, buffB);

  PrimFunc func(/*params=*/{}, loop_i, VoidType(), buffer_map);
  return IRModule::FromExpr(func);
}

void TirPartitionLoopTest() {

  LOG_SPLIT_LINE("TirPartitionLoopTest");
  auto mod = PartitionedLoop();
  LOG_PRINT_VAR(mod);  /// output:
  // mod: # from tvm.script import ir as I
  // # from tvm.script import tir as T
  // @I.ir_module
  // class Module:
  //     @T.prim_func(private=True)
  //     def main():
  //         A = T.Buffer((40,), "int32")
  //         B = T.Buffer((40,), "int32")
  //         for i, j in T.grid(4, 10):
  //             if i * 10 + j < 36:
  //                 A[10 * i + j] = B[10 * i + j]

  /// (partition_const_loop: true)
  Map<tvm::String, ObjectRef> inner_config;
  inner_config.Set("partition_const_loop", Bool(true));
  inner_config.Set("no_unroll_loop_with_extent_one", Bool(false));
  inner_config.Set("unroll_loop_with_partition_hint_no_interval", Bool(false));

  /// (tir.LoopPartition: inner_config)
  Map<String, ObjectRef> pass_config;
  pass_config.Set("tir.LoopPartition", inner_config);

  PassContext pass_ctx = PassContext::Current();
  pass_ctx->config = pass_config;

  {
    tvm::With<PassContext> scope(pass_ctx);

    PassContext current = PassContext::Current();
    LOG(INFO) << "Current PassContext: " << current;

    if (!current.defined()) {
      LOG(FATAL) << "PassContext is NULL!";
    }

    auto pass = LoopPartition();
    LOG(INFO) << "Creating LoopPartition pass";

    /// @bug @BenkangPeng Aborted(core dumped)
    /// dump at tir/ir/transform.cc line121: func = pass_func(std::move(func), mod,
    /// pass_ctx);
    // auto partitionedMod = pass(mod);

    // LOG(INFO) << "Pass applied successfully";

    // LOG_PRINT_VAR(partitionedMod);  // the same as mod, no change
  }
}

}  // namespace transform_test

REGISTER_TEST_SUITE(transform_test::TirVectorizeLoopTest,
                    tir_transform_test_TirVectorizeLoopTest);
REGISTER_TEST_SUITE(transform_test::TirPartitionLoopTest,
                    tir_transform_test_TirPartitionLoopTest);
