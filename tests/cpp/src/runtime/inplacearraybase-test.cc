#include "runtime/inplacearraybase-test.h"
#include "test-func-registry.h"

using namespace tvm::runtime;

void InplaceArrayBaseTest() {
  typedef uint32_t ArrayType;
  typedef int16_t ElemType;

  MyArray<ArrayType, ElemType> arr(10, 25, 49, 58, 890);
  arr.Show(std::cout);

  typedef double ElemType2;
  MyArray<ArrayType, ElemType2> arr2(10., 25., 49., 58., 890.);
  arr2.Show(std::cout);
}

namespace {

REGISTER_TEST_SUITE(InplaceArrayBaseTest);

}
