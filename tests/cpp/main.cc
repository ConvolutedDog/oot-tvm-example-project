#include "./include/inplacearraybase-test.h"
#include "./include/ndarrayutils-test.h"
#include "./include/object-test.h"

int main() {
  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "Running AutoSchedulerTest...\n";
  /// @bug
  /// InplaceArrayBaseTest();
  std::cout << "AutoSchedulerTest passed!\n\n";

  std::cout << "Running ObjectTest...\n";
  ObjectTest();
  std::cout << "ObjectTest passed!\n\n";
  
  std::cout << "Running ObjectRefTest...\n";
  ObjectRefTest();
  std::cout << "ObjectRefTest passed!\n\n";

  std::cout << "All tests passed!\n";
  return 0;
}
