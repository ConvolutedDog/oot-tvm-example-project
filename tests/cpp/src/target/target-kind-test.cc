#include "target/target-kind-test.h"
#include <tvm/runtime/logging.h>

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace target_kind_test {

/// @brief Please refer to the op-test.h for more details about `TargetKindRegEntry`.
void TargetKindTest() {
  LOG_SPLIT_LINE("TargetKindTest");

  Array<String> targetkindnames = TargetKindRegEntry::ListTargetKinds();
  for (const auto &name : targetkindnames) {
    // Only test llvm and cuda
    if (name == "llvm" || name == "cuda" || name == "nvptx") {
      LOG_SPLIT_LINE("TargetKindName: " + name);
      // TargetKindRegEntry targetkindentry = TargetKindRegEntry::RegisterOrGet(name);
      TargetKind targetkind = TargetKind::Get(name).value();
      Map<String, String> targetkindoptions =
          TargetKindRegEntry::ListTargetKindOptions(targetkind);
      for (const auto &option : targetkindoptions) {
        LOG_PRINT_VAR(" " + option.first + ": " + option.second);
      }
      LOG_PRINT_VAR(targetkind->name);
      LOG_PRINT_VAR(DLDeviceType2Str(targetkind->default_device_type));
      LOG_PRINT_VAR(targetkind->default_keys);
      LOG_PRINT_VAR(targetkind->preprocessor);
      LOG_PRINT_VAR(targetkind->target_parser);
      /// @brief All of the current TargetKinds registered in the TargetKindRegEntry don't
      /// have set attributes.
      /// LOG_PRINT_VAR(targetkind.GetAttrMap<Array<String>>("attribute-name")[targetkind]);
    }
  }
}

}  // namespace target_kind_test

void TargetKindTest() { target_kind_test::TargetKindTest(); }
