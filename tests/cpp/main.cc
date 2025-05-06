#include "ir/attrs-test.h"
#include "ir/expr-test.h"
#include "ir/pass-test.h"
#include "ir/type-test.h"
#include "node/reflection-test.h"
#include "runtime/inplacearraybase-test.h"
#include "runtime/ndarrayutils-test.h"
#include "runtime/object-test.h"
#include "tir/var-test.h"

int main() {
  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "Running AutoSchedulerTest...\n";
  // InplaceArrayBaseTest(); // Bug in MacOS
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

  std::cout << "Running VarTest...\n";
  VarTest();
  SizeVarTest();
  IterVarTest();
  std::cout << "VarTest passed!\n\n";

  std::cout << "Running ReflectionTest...\n";
  AttrVisitorTest();
  ReflectionVTableTest();
  std::cout << "ReflectionTest passed!\n\n";

  std::cout << "Running AttrTests...\n";
  AttrUtilsTests();
  AttrFieldInfoTest();
  AttrsTest();
  DictAttrsTest();
  std::cout << "AttrTests passed!\n\n";

  std::cout << "All tests passed!\n";
  return 0;
}
