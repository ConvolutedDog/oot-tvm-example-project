#include "tir/transform-test.h"
#include "test-func-registry.h"
#include "tvm/tir/op.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/with.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>
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
  Var i("i"), j("j"), k("k");

  Var m("m"), n("n");

  Stmt loop_k = For(k, 0, m, ForKind::kSerial,
                    IfThenElse(likely(i * m + j + k < n), Evaluate(m), Evaluate(n)));
  Stmt loop_j = For(j, 0, n, ForKind::kSerial, loop_k);
  Stmt loop_i = For(i, 0, 4, ForKind::kSerial, loop_j);

  PrimFunc func(/*params=*/{m, n}, loop_i, VoidType());
  return IRModule::FromExpr(func);
}

void TirPartitionLoopTest() {

  LOG_SPLIT_LINE("TirPartitionLoopTest");
  auto mod = PartitionedLoop();
  LOG_PRINT_VAR(mod);  /// output:

  Map<String, ObjectRef> pass_config_str;
  pass_config_str.Set("partition_const_loop", tvm::Bool(true));

  auto *reflection = ReflectionVTable::Global();

  /// @brief Legalize the pass config, which converts the `Map<String,Bool>` to
  /// `LoopPartitionConfig`.
  /// @param obj The pass config, which is a `Map<String,Bool>`.
  /// @param type_key struct `LoopPartitionConfigNode`'s `_type_key`.
  /// @return The legalized pass config, which is a `LoopPartitionConfig`.
  /// @sa tvm/src/tir/transforms/loop_partition.cc
  auto legalization = [=](const ObjectRef &obj) -> ObjectRef {
    return reflection->CreateObject("tir.transform.LoopPartitionConfig",
                                    Downcast<Map<String, ObjectRef>>(obj));
  };

  auto pass_config = legalization(pass_config_str);

  Map<String, ObjectRef> config;
  config.Set("tir.LoopPartition", pass_config);

  PassContext pass_ctx = PassContext::Create();
  pass_ctx->config = config;

  {
    With<PassContext> scope(pass_ctx);
    IRModule partitionedMod = LoopPartition()(mod);
    LOG_PRINT_VAR(partitionedMod);
  }
}

void TirUnrollLoopTest() {
  SizeVar n("n");
  Var i("i"), j("j");

  DataType dtype = DataType::Int(32, 1);
  Var bufPtr("bufferPtr", PointerType(PrimType(dtype), "global"));

  Buffer buf = Buffer(bufPtr, dtype, {n}, {1}, 0, "buffer", 4, 0, BufferType::kDefault);
  Stmt body = BufferStore(buf, BufferLoad(buf, {i}) + 1, {j + 1});

  Stmt loop_j = For(j, 0, 8, ForKind::kUnrolled, body);
  Stmt loop_i = For(i, n, (n + 2 - n), ForKind::kSerial, loop_j);

  Map<Var, Buffer> map;
  map.Set(bufPtr, buf);
  PrimFunc func({bufPtr}, loop_i, VoidType(), map);

  IRModule mod = IRModule::FromExpr(func);

  LOG_PRINT_VAR(mod);

  Map<String, ObjectRef> pass_config_str;
  pass_config_str.Set("auto_max_step", PrimExpr(16));

  auto *reflection = ReflectionVTable::Global();

  /// @brief Legalize the pass config, which converts the `Map<String,Bool>` to
  /// `UnrollLoopConfig`.
  /// @param obj The pass config, which is a `Map<String,Bool>`.
  /// @param type_key struct `UnrollLoopConfigNode`'s `_type_key`.
  /// @return The legalized pass config, which is a `UnrollLoopConfig`.
  /// @sa tvm/src/tir/transforms/unroll_loop.cc
  auto legalization = [=](const ObjectRef &obj) -> ObjectRef {
    return reflection->CreateObject("tir.transform.UnrollLoopConfig",
                                    Downcast<Map<String, ObjectRef>>(obj));
  };

  auto pass_config = legalization(pass_config_str);

  Map<String, ObjectRef> config;
  config.Set("tir.UnrollLoop", pass_config);

  PassContext pass_ctx = PassContext::Create();
  pass_ctx->config = config;

  {
    With<PassContext> scope(pass_ctx);
    IRModule unrollMod = UnrollLoop()(mod);
    LOG_PRINT_VAR(unrollMod);
  }
}

}  // namespace transform_test

REGISTER_TEST_SUITE(transform_test::TirVectorizeLoopTest,
                    tir_transform_test_TirVectorizeLoopTest);
REGISTER_TEST_SUITE(transform_test::TirPartitionLoopTest,
                    tir_transform_test_TirPartitionLoopTest);
REGISTER_TEST_SUITE(transform_test::TirUnrollLoopTest,
                    tir_transform_test_TirUnrollLoopTest);
