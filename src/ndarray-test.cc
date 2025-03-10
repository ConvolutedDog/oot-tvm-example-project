#include "ndarray-test.h"
#include <iostream>

namespace tvm {
namespace runtime {
void NDArrayTest() {
  // Create a ShapeTuple with initial dimensions {3, 64, 224, 224}.
  ShapeTuple shapetuple({3, 64, 224, 224});

  // Reassign ShapeTuple with new dimensions {2, 3, 4, 5}.
  shapetuple = std::vector<ShapeTuple::index_type>({2, 3, 4, 5});

  // Create an empty NDArray with the specified shape, data type, and device.
  NDArray ndarray =
      NDArray::Empty(shapetuple, DLDataType({2, 16, 1}), Device({kDLCPU, 0}));

  // Print the use count of the NDArray (reference count for shared_ptr).
  std::cout << "ndarray.use_count(): " << ndarray.use_count() << '\n';

  // Get the underlying DLTensor pointer from the NDArray.
  const DLTensor *dltensor = ndarray.operator->();

  // Get the shape of the NDArray and assign it to ShapeTuple.
  shapetuple = ndarray.Shape();

  // Print the number of dimensions of the DLTensor.
  std::cout << "dltensor->ndim: " << shapetuple.size() << '\n';

  // Get a pointer to the shape data and print each dimension.
  const ShapeTuple::index_type *index_data = shapetuple.data();
  for (size_t i = 0; i < shapetuple.size(); ++i) {
    std::cout << "index_data[" << i << "]: " << index_data[i] << '\n';
    std::cout << "  index_data @ " << i << ": " << shapetuple.at(i) << '\n';
  }

  // Print the first and last elements of the shape data.
  std::cout << "index_data front: " << shapetuple.front() << '\n';
  std::cout << "index_data back: " << shapetuple.back() << '\n';

  // Get the ShapeTupleObj pointer and print the product of its dimensions.
  const ShapeTupleObj *shapetupelobj = shapetuple.get();
  std::cout << "shapetupelobj->product: " << shapetupelobj->Product() << '\n';

  // Print the ShapeTuple object directly.
  std::cout << "shapetuple: " << shapetuple << '\n';

  // Check if the NDArray is contiguous in memory and print the result.
  std::cout << "ndarray.IsContiguous(): " << ndarray.IsContiguous() << '\n';

  // Create a new DLTensor and initialize it with data from the NDArray.
  DLTensor dltensor2;
  dltensor2.data =
      const_cast<void *>(reinterpret_cast<const void *>(ndarray.get()));
  dltensor2.device = Device({kDLCPU, 0});
  dltensor2.ndim = 4;
  dltensor2.dtype = DLDataType({2, 16, 1});
  dltensor2.shape = const_cast<int64_t *>(
      shapetuple.data()); // Convert const pointer to non-const.
  dltensor2.strides = nullptr;
  dltensor2.byte_offset = 0;

  // Create another empty NDArray with a different shape.
  NDArray ndarray2 = NDArray::Empty(
      ShapeTuple({3, 2, 5, 4}), DLDataType({2, 16, 1}), Device({kDLCPU, 0}));

  // Copy data from dltensor2 to ndarray2.
  ndarray2.CopyFrom(&dltensor2);

  // Print the shape of the new NDArray.
  std::cout << "ndarray2.Shape(): " << ndarray2.Shape() << '\n';
}

} // namespace runtime
} // namespace tvm
