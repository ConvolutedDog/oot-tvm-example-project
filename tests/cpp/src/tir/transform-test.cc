#include "tir/transform-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/container/map.h>
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

}  // namespace transform_test

REGISTER_TEST_SUITE(transform_test::TirVectorizeLoopTest,
                    tir_transform_test_TirVectorizeLoopTest);
