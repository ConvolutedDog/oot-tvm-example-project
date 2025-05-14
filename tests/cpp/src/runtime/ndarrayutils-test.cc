#include "runtime/ndarrayutils-test.h"
#include "test-func-registry.h"
#include <tvm/runtime/container/shape_tuple.h>

namespace tvm::runtime {

/// This function tests the NDArray API in TVM runtime.
/// It demonstrates how to create, manipulate, and inspect NDArray objects.
///
/// Key functionalities tested:
/// - Creation of NDArray objects.
/// - Printing of NDArray data based on its type.
/// - Shape and data type inspection.
/// - Copying data between NDArrays.
void NDArrayTest() {
  // Create a ShapeTuple with initial dimensions {3, 64, 224, 224}.
  ShapeTuple shapeTuple({3, 64, 224, 224});

  // Reassign ShapeTuple with new dimensions {2, 3, 4, 5}.
  shapeTuple = std::vector<ShapeTuple::index_type>({2, 3, 4, 5});

  // Create an empty NDArray with the specified shape, data type, and device.
  const NDArray ndarray =
      NDArray::Empty(shapeTuple, DLDataType({2, 16, 1}), Device({kDLCPU, 0}));

  // Print the use count of the NDArray (reference count for shared_ptr).
  std::cout << "ndarray.use_count(): " << ndarray.use_count() << '\n';

  // Get the underlying DLTensor pointer from the NDArray.
  const DLTensor *dltensor = ndarray.operator->();
  assert(dltensor != nullptr);

  // Get the shape of the NDArray and assign it to ShapeTuple.
  shapeTuple = ndarray.Shape();

  // Print the number of dimensions of the DLTensor.
  std::cout << "dltensor->ndim: " << shapeTuple.size() << '\n';

  // Get a pointer to the shape data and print each dimension.
  const ShapeTuple::index_type *indexData = shapeTuple.data();
  for (size_t i = 0; i < shapeTuple.size(); ++i) {
    std::cout << "indexData[" << i << "]: " << indexData[i] << '\n';
    std::cout << "  indexData @ " << i << ": " << shapeTuple.at(i) << '\n';
  }

  // Print the first and last elements of the shape data.
  std::cout << "indexData front: " << shapeTuple.front() << '\n';
  std::cout << "indexData back: " << shapeTuple.back() << '\n';

  // Get the ShapeTupleObj pointer and print the product of its dimensions.
  const ShapeTupleObj *shapeTupleObj = shapeTuple.get();
  std::cout << "shapeTupleObj->product: " << shapeTupleObj->Product() << '\n';

  // Print the ShapeTuple object directly.
  std::cout << "shapeTuple: " << shapeTuple << '\n';

  // Check if the NDArray is contiguous in memory and print the result.
  std::cout << "ndarray.IsContiguous(): " << ndarray.IsContiguous() << '\n';

  // Create a new DLTensor and initialize it with data from the NDArray.
  DLTensor dltensor2;
  dltensor2.data = const_cast<void *>(reinterpret_cast<const void *>(ndarray.get()));
  dltensor2.device = Device({kDLCPU, 0});
  dltensor2.ndim = 4;
  dltensor2.dtype = DLDataType({2, 16, 1});
  dltensor2.shape =
      const_cast<int64_t *>(shapeTuple.data());  // Convert const pointer to non-const.
  dltensor2.strides = nullptr;
  dltensor2.byte_offset = 0;

  // Create another empty NDArray with a different shape.
  NDArray ndarray2 = NDArray::Empty(ShapeTuple({3, 2, 5, 4}), DLDataType({2, 16, 1}),
                                    Device({kDLCPU, 0}));

  // Copy data from dltensor2 to ndarray2.
  ndarray2.CopyFrom(&dltensor2);

  // Print the shape of the new NDArray.
  std::cout << "ndarray2.Shape(): " << ndarray2.Shape() << '\n';

  // Copy data from the original NDArray to ndarray2.
  ndarray2.CopyFrom(ndarray);
  std::cout << "ndarray2.Shape(): " << ndarray2.Shape() << '\n';

  // Copy raw bytes from dltensor2 to ndarray2.
  int32_t numel = 1;
  for (size_t i = 0; i < dltensor2.ndim; ++i)
    numel *= dltensor2.shape[i];
  ndarray2.CopyFromBytes(dltensor2.data, dltensor2.dtype.bits / 8 * numel);
  std::cout << "ndarray2.Shape(): " << ndarray2.Shape() << '\n';

  // Create a third NDArray and copy data from ndarray2 to it.
  const NDArray ndarray3 = NDArray::Empty(ShapeTuple({5, 4, 3, 2}),
                                          DLDataType({1, 16, 1}), Device({kDLCPU, 0}));
  ndarray2.CopyTo(ndarray3);
  std::cout << "ndarray3.Shape(): " << ndarray3.Shape() << '\n';

  // Create an NDArrayWithPrinter object and print its details.
  NDArrayWithPrinter ndarray4(ndarray3);
  std::cout << "ndarray4.Shape(): " << ndarray4.Shape() << '\n';
  std::cout << "ndarray4.dtype: " << ndarray4.DataType2String() << '\n';
  std::cout << "ndarray4.data: " << '\n';
  ndarray4.Show();
}

}  // namespace tvm::runtime

void NDArrayTest() { tvm::runtime::NDArrayTest(); }

namespace {

REGISTER_TEST_SUITE(NDArrayTest);

}
