#include "tvm/relax/type.h"
#include <tvm/runtime/data_type.h>

namespace type_test {

using tvm::relax::ObjectType;
using tvm::relax::PackedFuncType;
using tvm::relax::ShapeType;
using tvm::relax::TensorType;

using tvm::DataType;
using tvm::relax::kUnknownNDim;

void RelaxShapeTypeTest();
void RelaxTensorTypeTest();
void RelaxObjectTypeTest();
void RelaxPackedFuncTypeTest();

}  // namespace type_test
