#include "runtime/inplacearraybase-test.h"
#include "test-func-registry.h"

namespace inplace_array_base_test {

using namespace tvm::runtime;

void RuntimeInplaceArrayBaseTest() {
  typedef uint32_t ArrayType;
  typedef int16_t ElemType;

#ifndef __APPLE__
  MyArray<ArrayType, ElemType> arr(10, 25, 49, 58, 890);
  arr.Show(std::cout);

  typedef double ElemType2;
  MyArray<ArrayType, ElemType2> arr2(10., 25., 49., 58., 890.);
  arr2.Show(std::cout);
#endif
}

}  // namespace inplace_array_base_test

REGISTER_TEST_SUITE(inplace_array_base_test::RuntimeInplaceArrayBaseTest,
                    runtime_inplace_array_base_test_RuntimeInplaceArrayBaseTest);
