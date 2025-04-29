#include "./include/expr-test.h"
#include "./include/inplacearraybase-test.h"
#include "./include/ndarrayutils-test.h"
#include "./include/object-test.h"
#include "./include/pass-test.h"
#include "./include/type-test.h"

int main() {
  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "Running AutoSchedulerTest...\n";
  InplaceArrayBaseTest();
  std::cout << "AutoSchedulerTest passed!\n\n";

  std::cout << "Running ObjectTest...\n";
  ObjectTest();
  std::cout << "ObjectTest passed!\n\n";

  std::cout << "Running ObjectRefTest...\n";
  ObjectRefTest();
  std::cout << "ObjectRefTest passed!\n\n";

  std::cout << "Running PassTest...\n";
  PassTest();
  std::cout << "PassTest passed!\n\n";

  std::cout << "Running ExprTest...\n";
  PrimExprTest();
  BoolTest();
  IntegerTest();
  RangeTest();
  std::cout << "ExprTest passed!\n\n";

  std::cout << "Running TypeTest...\n";
  PrimTypeTest();
  PointerTypeTest();
  TupleTypeTest();
  FuncTypeTest();
  std::cout << "TypeTest passed!\n\n";

  std::cout << "All tests passed!\n";
  return 0;
}
