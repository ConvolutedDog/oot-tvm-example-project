#include "tvm/ir/global_info.h"

namespace global_info_test {

using tvm::DummyGlobalInfo;
using tvm::GlobalInfo;
using tvm::VDevice;

using tvm::Target;
using tvm::TargetKind;

}  // namespace global_info_test

void GlobalInfoTest();
void VDeviceTest();
void DummyGlobalInfoTest();
