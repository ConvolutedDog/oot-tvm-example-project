#include "ir/global-info-test.h"
#include "test-func-registry.h"

namespace global_info_test {

void IrGlobalInfoTest() {}

void IrVDeviceTest() {
  LOG_SPLIT_LINE("IrVDeviceTest");

  Target target{"vulkan"};
  VDevice vd{target, 0, "memorycopeofvulkan"};
  LOG_PRINT_VAR(vd);
}

void IrDummyGlobalInfoTest() {}

}  // namespace global_info_test

REGISTER_TEST_SUITE(global_info_test::IrGlobalInfoTest,
                    ir_global_info_test_IrGlobalInfoTest);
REGISTER_TEST_SUITE(global_info_test::IrVDeviceTest, ir_global_info_test_IrVDeviceTest);
REGISTER_TEST_SUITE(global_info_test::IrDummyGlobalInfoTest,
                    ir_global_info_test_IrDummyGlobalInfoTest);
