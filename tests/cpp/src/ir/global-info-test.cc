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

void GlobalInfoTest() { global_info_test::GlobalInfoTest(); }
void VDeviceTest() { global_info_test::VDeviceTest(); }
void DummyGlobalInfoTest() { global_info_test::DummyGlobalInfoTest(); }

namespace {

REGISTER_TEST_SUITE(GlobalInfoTest);
REGISTER_TEST_SUITE(VDeviceTest);
REGISTER_TEST_SUITE(DummyGlobalInfoTest);

}  // namespace
