#include "te/tensor-test.h"
#include "test-func-registry.h"

namespace tensor_test {

void TeTensorTest() {
  LOG_SPLIT_LINE("TeTensorTest");

  IterVar h{
      {0, 223},
      Var{"h"},
      IterVarType::kUnrolled
  };
  IterVar w{
      {0, 223},
      Var{"w"},
      IterVarType::kUnrolled
  };
  ComputeOp computeop{
      "compute", "tag", {},
        {h, w},
        {tvm::tir::Add(h, w)}
  };
  LOG_PRINT_VAR(computeop);
  /// Output:
  ///   compute(compute, body=[h + w], axis=[
  ///           T.iter_var(h, T.Range(0, 223), "Unrolled", ""),
  ///           T.iter_var(w, T.Range(0, 223), "Unrolled", "")],
  ///           reduce_axis=[], tag=tag, attrs={})

  Tensor tensor{
      {16, 3, 224, 224},
      DataType::Float(32, 4), computeop, 0
  };
  LOG_PRINT_VAR(tensor);
  /// Output:
  ///   Tensor(shape=[16, 3, 224, 224], op.name=compute)

  LOG_PRINT_VAR(tensor.ndim());
  LOG_PRINT_VAR(tensor.operator()<Array<PrimExpr>>({1, 2, 112, 112}));
  Var indices1{"n"}, indices2{"c"}, indices3{"h"}, indices4{"w"};
  LOG_PRINT_VAR(
      tensor.operator()<Array<PrimExpr>>({indices1, indices2, indices3, indices4}));
  // LOG_PRINT_VAR(PrimExpr(tensor[indices1]));
  // LOG_PRINT_VAR(tensor[indices1][indices2]);
  // LOG_PRINT_VAR(tensor[indices1][indices2][indices3]);
  LOG_PRINT_VAR(PrimExpr(tensor[indices1][indices2][indices3][indices4]));

  Tensor tensor2 = Tensor(tensor);

  LOG_PRINT_VAR(tensor2 == tensor);
  LOG_PRINT_VAR(tensor2 != tensor);
  // LOG_PRINT_VAR(!tensor2[0][0][0][0]); // Require arg.dtype().is_bool()
  // LOG_PRINT_VAR(-tensor2[0][0][0][0]); // Require arg.dtype().is_bool()
  LOG_PRINT_VAR(tensor2[0][0][0][0] + tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] - tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] * tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] == tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] <= tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] >= tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] != tensor[0][0][0][0]);

  // Require arg.dtype().is_bool()
  // LOG_PRINT_VAR(tensor2[0][0][0][0] && tensor[0][0][0][0]);

  // Require arg.dtype().is_bool()
  // LOG_PRINT_VAR(tensor2[0][0][0][0] || tensor[0][0][0][0]);

  // Require lhs.dtype().is_int() || lhs.dtype().is_uint()
  // LOG_PRINT_VAR(tensor2[0][0][0][0] >> tensor[0][0][0][0]);

  // Require lhs.dtype().is_int() || lhs.dtype().is_uint()
  // LOG_PRINT_VAR(tensor2[0][0][0][0] << tensor[0][0][0][0]);
  
  LOG_PRINT_VAR(tensor2[0][0][0][0] > tensor[0][0][0][0]);
  LOG_PRINT_VAR(tensor2[0][0][0][0] < tensor[0][0][0][0]);
}

}  // namespace tensor_test

REGISTER_TEST_SUITE(tensor_test::TeTensorTest, te_tensor_test_TeTensorTest);
