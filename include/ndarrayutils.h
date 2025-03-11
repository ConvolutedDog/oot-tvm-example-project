
#include <dlpack/dlpack.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>

#ifndef NDARRAY_TEST_H
#define NDARRAY_TEST_H

/// A macro for assertions with a custom error message.
/// If the condition is false, the macro prints an error message and triggers
/// an assertion failure.
///
/// @param condition The condition to be checked.
/// @param message A custom error message to be printed if the condition fails.
///
/// Usage:
/// ```c++
/// ASSERT_WITH_MSG(x > 0, "x must be positive");
/// ```
#define ASSERT_WITH_MSG(condition, message)                                    \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion failed: " << #condition << ", "                  \
                << "message: " << message << "\n";                             \
      assert(false);                                                           \
    }                                                                          \
  } while (false)

namespace tvm::runtime {

/// A template class for a printer that uses the CRTP (Curiously Recurring
/// Template Pattern). Derived classes must implement the `PrintImpl` method.
/// The `PrintImpl` method should not have a return value, as it will be
/// ignored. This utility is primarily intended for debugging and runtime
/// inspection.
///
/// @tparam Derived The derived class that implements the `PrintImpl` method.
///
/// Usage:
/// - Extend the `NDArrayPrinter` class by implementing the `PrintImpl` method
/// in a derived class.
/// - Use the `Show()` method to trigger the custom printing logic.
///
/// Example:
/// ```c++
/// class MyNDArray : public NDArrayPrinter<MyNDArray> {
/// public:
///   void PrintImpl() {
///     std::cout << "Custom printing logic" << std::endl;
///   }
/// };
/// ```
template <typename Derived> class NDArrayPrinter {
public:
  using index_type = ShapeTuple::index_type;

  /// Calls the `PrintImpl` method of the derived class.
  void Show();

  /// Helper function to print data of a specific type.
  ///
  /// @tparam T The type of data to be printed (e.g., int8_t, float).
  /// @param data Pointer to the raw data.
  /// @param size Number of elements in the data.
  /// @param type_name A string describing the data type (e.g., "Int8").
  template <typename T>
  void PrintData(void *data, size_t size, const std::string &typeName);

  /// Converts the DataType of the NDArray to a string representation.
  ///
  /// @return A string representation of the DataType (e.g., "int32").
  std::string DataType2String();

  /// Default constructor.
  NDArrayPrinter() = default;

  /// Declare the destructor as virtual to ensure proper cleanup in case of
  /// inheritance.
  virtual ~NDArrayPrinter() = default;
};

template <typename Derived> void NDArrayPrinter<Derived>::Show() {
  static_cast<Derived *>(this)->PrintImpl();
}

template <typename Derived>
template <typename T>
void NDArrayPrinter<Derived>::PrintData(void *data, size_t size,
                                        const std::string &typeName) {
  std::cout << typeName << " - NDArray data: ";
  T *ptr = static_cast<T *>(data);
  for (size_t i = 0; i < size; ++i) {
    std::cout << ptr[i] << ",";
  }
  std::cout << std::endl;
}

template <typename Derived>
std::string NDArrayPrinter<Derived>::DataType2String() {
  const auto *self = static_cast<Derived *>(this);
  return DLDataType2String(
      DLDataType({static_cast<uint8_t>(self->DataType().code()),
                  static_cast<uint8_t>(self->DataType().bits()),
                  static_cast<uint16_t>(self->DataType().lanes())}));
}

/// A class that combines NDArray functionality with a printer. Inherits from
/// both NDArray and NDArrayPrinter to enable printing of NDArray data.
class NDArrayWithPrinter : public NDArray,
                           public NDArrayPrinter<NDArrayWithPrinter> {
public:
  /// Default constructor.
  NDArrayWithPrinter() : NDArray() {}

  /// Constructor that initializes the NDArrayWithPrinter with an existing
  /// NDArray object.
  ///
  /// @param data A pointer to the underlying TVM Object.
  NDArrayWithPrinter(ObjectPtr<Object> data) : NDArray(std::move(data)) {}

  /// Constructor that initializes the NDArrayWithPrinter with an existing
  /// NDArray object.
  ///
  /// @param other An existing NDArray object.
  NDArrayWithPrinter(const NDArray &other) : NDArray(other) {}

private:
  /// Calculates the total number of elements in the NDArray.
  ///
  /// @return The total number of elements (product of all dimensions).
  ShapeTuple::index_type numel() const;

public:
  /// Implements the `PrintImpl` method required by the NDArrayPrinter class.
  /// Prints the data stored in the NDArray based on its data type.
  void PrintImpl();
};

}  // namespace tvm::runtime

#endif
