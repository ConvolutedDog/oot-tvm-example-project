#include "ir/global-info-test.h"
#include "test-func-registry.h"

namespace global_info_test {

void GlobalInfoTest() {}

void VDeviceTest() {
  LOG_SPLIT_LINE("VDeviceTest");

  Target target{"vulkan"};
  VDevice vd{target, 0, "memorycopeofvulkan"};
  LOG_PRINT_VAR(vd);
}

void DummyGlobalInfoTest() {}

}  // namespace global_info_test

REGISTER_TEST_SUITE(global_info_test::GlobalInfoTest, ir_global_info_test_GlobalInfoTest);
REGISTER_TEST_SUITE(global_info_test::VDeviceTest, ir_global_info_test_VDeviceTest);
REGISTER_TEST_SUITE(global_info_test::DummyGlobalInfoTest,
                    ir_global_info_test_DummyGlobalInfoTest);
