#include "tvm/runtime/packed_func.h"
#include "tvm/target/target.h"
#include "tvm/target/target_kind.h"

namespace target_test {

using tvm::CheckAndUpdateHostConsistency;
using tvm::Target;
using tvm::TargetKind;
using tvm::TargetKindRegEntry;

using tvm::runtime::Array;
using tvm::runtime::Map;
using tvm::runtime::String;

using tvm::runtime::DLDeviceType2Str;

void TargetTest();

}  // namespace target_test
