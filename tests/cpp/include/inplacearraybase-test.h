#include <tvm/runtime/container/base.h>

#ifndef INPLACEARRAYBASE_TEST_H
#define INPLACEARRAYBASE_TEST_H

namespace tvm::runtime {

template <typename Derived> class MyArrayPrinter {
public:
  void Show(std::ostream *os = &std::cout) {
    const auto &self = static_cast<const Derived &>(*this);
    *os << "MyArray: ";
    for (size_t i = 0; i < self.GetSize(); ++i) {
      *os << self[i] << ",";
    }
    *os << std::endl;
  }
};

template <typename ArrayType, typename ElemType>
class MyArray : public InplaceArrayBase<MyArray<ArrayType, ElemType>, ElemType>,
                public MyArrayPrinter<MyArray<ArrayType, ElemType>> {
public:
  using Base = InplaceArrayBase<MyArray<ArrayType, ElemType>, ElemType>;

  template <typename... Args> MyArray(Args &&...args) {
    size = sizeof...(args);
    this->InitMyArray(0, std::forward<Args>(args)...);
  }

  ~MyArray() = default;

  size_t GetSize() const { return size; }

private:
  size_t size;

  template <typename Arg, typename... Args>
  void InitMyArray(size_t index, Arg &&arg, Args &&...args) {
    this->template EmplaceInit<Arg>(index, std::forward<Arg>(arg));
    InitMyArray(index + 1, std::forward<Args>(args)...);
  }

  template <typename Arg> void InitMyArray(size_t index, Arg &&arg) {
    this->template EmplaceInit<Arg>(index, std::forward<Arg>(arg));
  }

  // Rewrite the function for debug.
  template <typename... Args> void EmplaceInit(size_t idx, Args &&...args) {
    std::cout << "this->AddressOf(idx): " << this->AddressOf(idx)
              << " sizeof(ElemType): " << sizeof(ElemType) << '\n';
    // NOLINTNEXTLINE(readability-identifier-naming)
    void *field_ptr = this->AddressOf(idx);
    new (field_ptr) ElemType(std::forward<Args>(args)...);
  }
};

}  // namespace tvm::runtime

void InplaceArrayBaseTest();

#endif
