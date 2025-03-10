#include <dlpack/dlpack.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>

namespace tvm {
namespace runtime {

template <typename Derived> class Printer {
public:
  void show() { static_cast<Derived *>(this)->PrintImpl(); }
};

class NDArrayWithPrinter : public NDArray, public Printer<NDArrayWithPrinter> {
public:
  NDArrayWithPrinter(const NDArray &other) : NDArray(other) {}

private:
  size_t Size() {
    size_t numel = 1;
    for (size_t i = 0; i < this->get_mutable()->dl_tensor.ndim; ++i)
      numel *= this->get_mutable()->dl_tensor.shape[i];
    return numel;
  }

public:
  void PrintImpl() {
    const ShapeTuple shapetupel = this->Shape();
    const ShapeTuple::index_type *shapedata = shapetupel.data();
    void *dltensordata = this->get_mutable()->dl_tensor.data;

    runtime::DataType type = this->DataType();
    if (type.code() == DataType::kInt || type.code() == DataType::kUInt) {
      if (type.bits() == 8) {
        std::cout << "Int8 NDArray data: ";
        for (int i = 0; i < this->Size(); ++i) {
          std::cout << static_cast<int8_t *>(dltensordata)[i] << " ";
        }
        std::cout << std::endl;
      } else if (type.bits() == 16) {
        std::cout << "Int16 NDArray data: ";
        for (int i = 0; i < this->Size(); ++i) {
          std::cout << static_cast<int16_t *>(dltensordata)[i] << " ";
        }
        std::cout << std::endl;
      } else if (type.bits() == 32) {
        std::cout << "Int32 NDArray data: ";
        for (int i = 0; i < this->Size(); ++i) {
          std::cout << static_cast<int32_t *>(dltensordata)[i] << " ";
        }
        std::cout << std::endl;
      }
    } else {
      std::cerr << "Unsupported data type: "
                << DLDataType2String(
                       DLDataType({static_cast<uint8_t>(type.code()),
                                   static_cast<uint8_t>(type.bits()),
                                   static_cast<uint16_t>(type.lanes())}))
                << std::endl;
    }
  }
};

// This function tests the NDArray API in TVM runtime.
// It demonstrates how to create, manipulate, and inspect NDArray objects.
void NDArrayTest();

} // namespace runtime
} // namespace tvm
