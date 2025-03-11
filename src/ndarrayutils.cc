#include "ndarrayutils.h"

namespace tvm::runtime {

/// The bit width of the data type.
typedef enum {
  kInt8 = 8,
  kInt16 = 16,
  kFloat16 = 16,
  kInt32 = 32,
  kFloat32 = 32,
  kFloat64 = 64,
} DLBitwidth;

/// Calculates the total number of elements in the NDArray.
///
/// @return The total number of elements (product of all dimensions).
/// @throws std::runtime_error If the NDArray has invalid dimensions.
ShapeTuple::index_type NDArrayWithPrinter::numel() const {
  int64_t ret = 1;
  auto &nDim = this->get_mutable()->dl_tensor.ndim;
  ASSERT_WITH_MSG(nDim >= 1,  // NOLINT(misc-static-assert)
                  "NDArray must have at least one dimension to compute numel.");
  for (int64_t i = 0; i < nDim; ++i) {
    auto &dim = this->get_mutable()->dl_tensor.shape[i];
    ASSERT_WITH_MSG(  // NOLINT(misc-static-assert)
        dim > 0, "NDArray must have positive dimensions to compute numel.\n");
    ret *= dim;
  }
  return ret;
}

/// Implements the `PrintImpl` method required by the NDArrayPrinter class.
/// Prints the data stored in the NDArray based on its data type.
///
/// @throws std::runtime_error If the data type is unsupported.
void NDArrayWithPrinter::PrintImpl() {
  // Get the shape of the NDArray.
  const ShapeTuple shapeTuple = this->Shape();

  // Get the raw data pointer from the DLTensor.
  void *dlTensorData = this->get_mutable()->dl_tensor.data;

  // Get the data type of the NDArray.
  const runtime::DataType type = this->DataType();

  // Get the total number of elements.
  const size_t numElements = this->numel();

  // Get the string representation of the NDArray's data type.
  const std::string nDarrayDtype = DataType2String();

  // Handle different data types.
  switch (type.code()) {
  case DataType::kInt:
  case DataType::kUInt:
    switch (type.bits()) {
    case kInt8: PrintData<int8_t>(dlTensorData, numElements, "Int8"); break;
    case kInt16: PrintData<int16_t>(dlTensorData, numElements, "Int16"); break;
    case kInt32: PrintData<int32_t>(dlTensorData, numElements, "Int32"); break;
    default:
      throw std::runtime_error("Unsupported integer bit width: " +
                               std::to_string(type.bits()));
    }
    break;
  case DataType::kFloat:
    switch (type.bits()) {
    case kFloat32: PrintData<float>(dlTensorData, numElements, "Float32"); break;
    case kFloat64: PrintData<double>(dlTensorData, numElements, "Float64"); break;
    default:
      throw std::runtime_error("Unsupported float bit width: " +
                               std::to_string(type.bits()));
    }
    break;
  default:
    throw std::runtime_error("Unsupported data type code: " + nDarrayDtype);
  }
}

}  // namespace tvm::runtime
