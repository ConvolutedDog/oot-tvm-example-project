#include "tir/function-test.h"
#include "test-func-registry.h"

namespace function_test {

void TirPrimFuncTest() {
  LOG_SPLIT_LINE("TirPrimFuncTest");

  /// PrimFuncNode contains TIR statements.
  Var m("m", DataType::Int(32));
  Var n("n", DataType::Int(32));
  int lanes = 2;
  DataType dtype = DataType::BFloat(16, lanes);
  TensorType retTy{2, dtype};
  Array<PrimExpr> shape{m, n};
  Tensor tensor{
      shape, dtype, PlaceholderOp{"placeholder", shape, dtype},
        0
  };
  PrimExpr value{0};
  Broadcast broadcast{value, lanes};
  Array<PrimExpr> indices = {m, n};
  ProducerStore producerstore{tensor, broadcast, indices};
  ProducerLoad producerload(tensor, indices);
  /// @bug The generated python code cannot execute.
  PrimFunc primfunc{
      {m, n},
      producerstore, retTy
  };
  LOG_PRINT_VAR(primfunc);
  /// # from tvm.script import tir as T
  /// # from tvm.script import relax as R
  ///
  /// @T.prim_func(private=True)
  /// def main(m: T.int32, n: T.int32) -> R.Tensor(ndim=2, dtype="bfloat16x2"):
  ///     placeholder[m, n] = T.Broadcast(0, 2)

  LOG_PRINT_VAR(primfunc->func_type_annotation());
  /// Output:
  ///   I.FuncType([T.int32, T.int32], R.Tensor(ndim=2, dtype="bfloat16x2"))

  LOG_PRINT_VAR(primfunc->params);         // [m, n]
  LOG_PRINT_VAR(primfunc->body);           // m = T.int32()
                                           // n = T.int32()
                                           // placeholder[m, n] = T.Broadcast(0, 2)
  LOG_PRINT_VAR(primfunc->ret_type);       // R.Tensor(ndim=2, dtype="bfloat16x2")
  LOG_PRINT_VAR(primfunc->buffer_map);     // {}
  LOG_PRINT_VAR(primfunc->checked_type_);  // Same to func_type_annotation()
  LOG_PRINT_VAR(primfunc->struct_info_);   // R.Callable((R.Prim("int32"),
                                           // R.Prim("int32")), R.Object, True)
}

/// @todo (yangjianchao)
void TirTensorIntrinTest() { LOG_SPLIT_LINE("TirTensorIntrinTest"); }

/// @todo (yangjianchao)
void TirSpecialize() { LOG_SPLIT_LINE("TirSpecialize"); }

}  // namespace function_test

REGISTER_TEST_SUITE(function_test::TirPrimFuncTest, tir_function_test_TirPrimFuncTest);
REGISTER_TEST_SUITE(function_test::TirTensorIntrinTest,
                    tir_function_test_TirTensorIntrinTest);
REGISTER_TEST_SUITE(function_test::TirSpecialize, tir_function_test_TirSpecialize);
