#include "include/ndarrayutils-test.h"
#include "include/inplacearraybase-test.h"

int main() {
  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "Running AutoSchedulerTest...\n";
  InplaceArrayBaseTest();
  std::cout << "AutoSchedulerTest passed!\n\n";

  std::cout << "All tests passed!\n";
  return 0;
}
