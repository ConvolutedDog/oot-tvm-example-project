#include "ndarrayutils.h"

namespace tvm {
namespace runtime {

/// Calculates the total number of elements in the NDArray.
///
/// @return The total number of elements (product of all dimensions).
/// @throws std::runtime_error If the NDArray has invalid dimensions.
ShapeTuple::index_type NDArrayWithPrinter::numel() const {
  int64_t ret = 1;
  auto &ndim = this->get_mutable()->dl_tensor.ndim;
  ASSERT_WITH_MSG(
      ndim >= 1,
      "NDArray must have at least one dimension to compute numel.");
  for (int64_t i = 0; i < this->get_mutable()->dl_tensor.ndim; ++i) {
    auto &dim = this->get_mutable()->dl_tensor.shape[i];
    ASSERT_WITH_MSG(
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
  const ShapeTuple shape_tuple = this->Shape();

  // Get the raw data pointer from the DLTensor.
  void *dl_tensor_data = this->get_mutable()->dl_tensor.data;

  // Get the data type of the NDArray.
  runtime::DataType type = this->DataType();

  // Get the total number of elements.
  size_t num_elements = this->numel();

  // Get the string representation of the NDArray's data type.
  std::string ndarray_dtype = DataType2String();

  // Handle different data types.
  switch (type.code()) {
  case DataType::kInt:
  case DataType::kUInt:
    switch (type.bits()) {
    case 8:
      PrintData<int8_t>(dl_tensor_data, num_elements, "Int8");
      break;
    case 16:
      PrintData<int16_t>(dl_tensor_data, num_elements, "Int16");
      break;
    case 32:
      PrintData<int32_t>(dl_tensor_data, num_elements, "Int32");
      break;
    default:
      throw std::runtime_error("Unsupported integer bit width: " +
                               std::to_string(type.bits()));
    }
    break;
  case DataType::kFloat:
    switch (type.bits()) {
    case 32:
      PrintData<float>(dl_tensor_data, num_elements, "Float32");
      break;
    case 64:
      PrintData<double>(dl_tensor_data, num_elements, "Float64");
      break;
    default:
      throw std::runtime_error("Unsupported float bit width: " +
                               std::to_string(type.bits()));
    }
    break;
  default:
    throw std::runtime_error("Unsupported data type code: " + ndarray_dtype);
  }
}

} // namespace runtime
} // namespace tvm
