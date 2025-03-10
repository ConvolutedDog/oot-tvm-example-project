#include "include/example-test.h"
#include "include/ndarrayutils-test.h"

int main() {
  std::cout << "Running RecursiveExprRegression...\n";
  tvm::relay::RecursiveExprRegression();
  std::cout << "RecursiveExprRegression passed!\n\n";

  std::cout << "Running UnusedLetVars...\n";
  tvm::relay::UnusedLetVars();
  std::cout << "UnusedLetVars passed!\n\n";

  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "All tests passed!\n";
  return 0;
}
