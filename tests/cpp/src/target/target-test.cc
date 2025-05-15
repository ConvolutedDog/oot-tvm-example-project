#include "target/target-test.h"
#include "test-func-registry.h"
#include <tvm/runtime/logging.h>

namespace target_test {

std::ostream &operator<<(std::ostream &os, const std::vector<std::string> &vec) {
  os << "vector<string>{";
  for (auto &s : vec)
    os << s << ", ";
  os << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::unordered_set<std::string> &set) {
  os << "unordered_set<string>{";
  for (auto &s : set)
    os << s << ", ";
  os << "}";
  return os;
}

void TargetTest() {
  LOG_SPLIT_LINE("TargetTest");

  // Constructors
  Target target{"vulkan"};

  LOG_PRINT_VAR(target);
  LOG_PRINT_VAR(Target::Current());  // nullptr
  LOG_PRINT_VAR(target->kind);
  LOG_PRINT_VAR(target->host);
  LOG_PRINT_VAR(target->tag);
  LOG_PRINT_VAR(target->keys);
  LOG_PRINT_VAR(target->attrs);
  LOG_PRINT_VAR(target->features);

  LOG_PRINT_VAR(target->Export());
  LOG_PRINT_VAR(target->GetHost() == nullptr);
  LOG_PRINT_VAR(DLDeviceType2Str(target->GetTargetDeviceType()));
  LOG_PRINT_VAR(target->HasKey("vulkan"));
  LOG_PRINT_VAR(target->HasKey("gpu"));
  LOG_PRINT_VAR(target->ToDebugString());

  LOG_PRINT_VAR(target->GetAttr<String>("mcpu"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_float16"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_float32"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_float64"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_int8"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_int16"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_int32"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_int64"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_8bit_buffer"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_16bit_buffer"));
  LOG_PRINT_VAR(
      target->GetAttr<tvm::runtime::Bool>("supports_storage_buffer_storage_class"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_push_descriptor"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_dedicated_allocation"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Bool>("supports_integer_dot_product"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("supports_cooperative_matrix"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("supported_subgroup_operations"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_num_threads"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_threads_per_block"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("thread_warp_size"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_block_size_x"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_block_size_y"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_block_size_z"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_push_constants_size"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_uniform_buffer_range"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_storage_buffer_range"));
  LOG_PRINT_VAR(
      target->GetAttr<tvm::runtime::Int>("max_per_stage_descriptor_storage_buffer"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_shared_memory_per_block"));
  LOG_PRINT_VAR(target->GetAttr<String>("device_type"));
  LOG_PRINT_VAR(target->GetAttr<String>("device_name"));
  LOG_PRINT_VAR(target->GetAttr<String>("driver_name"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("driver_version"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("vulkan_api_version"));
  LOG_PRINT_VAR(target->GetAttr<tvm::runtime::Int>("max_spirv_version"));

  LOG_PRINT_VAR(target->GetKeys());
  LOG_PRINT_VAR(target->GetLibs());

  /// Test target features
  Target targettest{"test"};
  LOG_PRINT_VAR(targettest->features);
  LOG_PRINT_VAR(targettest->GetFeature<tvm::runtime::Bool>("is_test"));
  LOG_PRINT_VAR(targettest->GetKeys());
  LOG_PRINT_VAR(targettest->GetLibs());
}

}  // namespace target_test

REGISTER_TEST_SUITE(target_test::TargetTest, target_target_test_TargetTest);
