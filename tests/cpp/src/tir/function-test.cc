#include "tir/function-test.h"
#include "test-func-registry.h"

namespace function_test {

void TirPrimFuncTest() {
  LOG_SPLIT_LINE("TirPrimFuncTest");

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
