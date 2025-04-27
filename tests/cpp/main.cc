#include "./include/inplacearraybase-test.h"
#include "./include/ndarrayutils-test.h"
#include "./include/object-test.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << stmt << std::endl

int main() {
  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "Running AutoSchedulerTest...\n";
  /// @bug
  /// InplaceArrayBaseTest();
  std::cout << "AutoSchedulerTest passed!\n\n";

  object_test::TestCanDerivedFromObject testCanDerivedFromObj =
      object_test::InitObject<object_test::TestCanDerivedFromObject>();
  LOG_PRINT_VAR(testCanDerivedFromObj.type_index());
  LOG_PRINT_VAR(testCanDerivedFromObj.RuntimeTypeIndex());
  LOG_PRINT_VAR(testCanDerivedFromObj.GetTypeKey());
  LOG_PRINT_VAR(testCanDerivedFromObj.type_index());
  LOG_PRINT_VAR(testCanDerivedFromObj
                    .IsInstance<object_test::TestCanDerivedFromObject>());

  object_test::TestDerived testDerived =
      object_test::InitObject<object_test::TestDerived>();
  LOG_PRINT_VAR(testDerived.type_index());
  LOG_PRINT_VAR(testDerived.RuntimeTypeIndex());
  LOG_PRINT_VAR(testDerived.GetTypeKey());
  LOG_PRINT_VAR(testDerived.type_index());

  object_test::TestDerived1 testDerived1 =
      object_test::InitObject<object_test::TestDerived1>();
  LOG_PRINT_VAR(testDerived1.type_index());
  LOG_PRINT_VAR(testDerived1.RuntimeTypeIndex());
  LOG_PRINT_VAR(testDerived1.GetTypeKey());
  LOG_PRINT_VAR(testDerived1.type_index());

  object_test::TestFinalObject testFinalObj =
      object_test::InitObject<object_test::TestFinalObject>();
  LOG_PRINT_VAR(testFinalObj.type_index());
  LOG_PRINT_VAR(testFinalObj.RuntimeTypeIndex());
  LOG_PRINT_VAR(testFinalObj.GetTypeKey());
  LOG_PRINT_VAR(testFinalObj.type_index());

  std::cout << "All tests passed!\n";
  return 0;
}
