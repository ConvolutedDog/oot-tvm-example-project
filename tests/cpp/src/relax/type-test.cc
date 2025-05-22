#include "relax/type-test.h"
#include "test-func-registry.h"
#include <tvm/runtime/data_type.h>

namespace type_test {

void RelaxShapeTypeTest() {
  LOG_SPLIT_LINE("ShapeTypeTest");

  ShapeType shapetype{kUnknownNDim};
  LOG_PRINT_VAR(shapetype);

  shapetype = ShapeType{4};
  LOG_PRINT_VAR(shapetype);
}

void RelaxTensorTypeTest() {
  LOG_SPLIT_LINE("TensorTypeTest");

  TensorType tensortype{4, DataType::Float(32, 4)};
  LOG_PRINT_VAR(tensortype);                    // R.Tensor(ndim=4, dtype="float32x4")
  LOG_PRINT_VAR(tensortype->ndim);              // 4
  LOG_PRINT_VAR(tensortype->dtype);             // float32x4
  LOG_PRINT_VAR(tensortype->IsUnknownDtype());  // 0
  LOG_PRINT_VAR(tensortype->IsUnknownNdim());   // 0

  tensortype = TensorType{4, DataType::Handle()};
  LOG_PRINT_VAR(tensortype);                    // R.Tensor(ndim=4, dtype="handle")
  LOG_PRINT_VAR(tensortype->ndim);              // 4
  LOG_PRINT_VAR(tensortype->dtype);             // handle
  LOG_PRINT_VAR(tensortype->IsUnknownDtype());  // 0
  LOG_PRINT_VAR(tensortype->IsUnknownNdim());   // 0

  tensortype = TensorType{kUnknownNDim, DataType::Handle(0, 0)};
  LOG_PRINT_VAR(tensortype);                    // R.Tensor(ndim=-1, dtype="void")
  LOG_PRINT_VAR(tensortype->ndim);              // -1
  LOG_PRINT_VAR(tensortype->dtype);             // void
  LOG_PRINT_VAR(tensortype->IsUnknownDtype());  // 1
  LOG_PRINT_VAR(tensortype->IsUnknownNdim());   // 1

  tensortype = TensorType::CreateUnknownNDim(DataType::Handle(0, 0));
  LOG_PRINT_VAR(tensortype);                    // R.Tensor(ndim=-1, dtype="void")
  LOG_PRINT_VAR(tensortype->ndim);              // -1
  LOG_PRINT_VAR(tensortype->dtype);             // void
  LOG_PRINT_VAR(tensortype->IsUnknownDtype());  // 1
  LOG_PRINT_VAR(tensortype->IsUnknownNdim());   // 1
}

void RelaxObjectTypeTest() {
  LOG_SPLIT_LINE("ObjectTypeTest");

  ObjectType objecttype{};
  LOG_PRINT_VAR(objecttype);  // R.Object
}

void RelaxPackedFuncTypeTest() {
  LOG_SPLIT_LINE("PackedFuncTypeTest");

  PackedFuncType packedfunctype{};
  LOG_PRINT_VAR(packedfunctype);  // R.PackedFunc
}

}  // namespace type_test

REGISTER_TEST_SUITE(type_test::RelaxShapeTypeTest, relax_type_test_ShapeTypeTest);
REGISTER_TEST_SUITE(type_test::RelaxTensorTypeTest, relax_type_test_TensorTypeTest);
REGISTER_TEST_SUITE(type_test::RelaxObjectTypeTest, relax_type_test_ObjectTypeTest);
REGISTER_TEST_SUITE(type_test::RelaxPackedFuncTypeTest,
                    relax_type_test_PackedFuncTypeTest);
