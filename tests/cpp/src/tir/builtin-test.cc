#include "tir/builtin-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/optional.h>
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include "tvm/target/codegen.h"
#include "tvm/relax/transform.h"
#include "tvm/ir/transform.h"
#include "tvm/tir/transform.h"

namespace builtin_test {

void TirretTest() {
  LOG_SPLIT_LINE("TirretTest");

  auto &op = ret();
  LOG_PRINT_VAR(op->name);
  tvm::tir::Call call{tvm::DataType::Float(32), op, {tvm::PrimExpr{1}}};
  LOG_PRINT_VAR(call);
}

void TirreinterpretTest() {
  LOG_SPLIT_LINE("TirreinterpretTest");
  auto &op = reinterpret();
  LOG_PRINT_VAR(op->name);
  tvm::tir::Call call{tvm::DataType::Float(32), op, {tvm::PrimExpr{1}}};
}

void TirlikelyTest() {}

void Tirbitwise_andTest() {}

void Tirbitwise_orTest() {}

void Tirbitwise_xorTest() {}

void Tirbitwise_notTest() {}

void Tirshift_leftTest() {
  LOG_SPLIT_LINE("Tirshift_leftTest");

  auto &op = shift_left();
  tvm::GlobalVar globalvar("globalvar");

  /// Create tvm::relax::Function
  tvm::RelaxExpr opexpr = tvm::Op::Get("relax.add");
  tvm::relax::Var arg1("arg1", tvm::relax::TensorStructInfo(tvm::DataType::Float(32), 4));
  tvm::relax::Var arg2("arg2", tvm::relax::TensorStructInfo(tvm::DataType::Float(32), 4));
  tvm::relax::Call call{
      opexpr, {arg1, arg2}
  };
  tvm::relax::Function func{
      {arg1, arg2},
      call,
      tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4},
      true,
  };

  /// @note TVMScript cannot print functions of type: BaseFunc
  tvm::IRModule irmodule{{std::pair<tvm::GlobalVar, tvm::BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule);

  /// @todo
  // tvm::transform::Sequential passseq = tvm::transform::Sequential({
  //     tvm::relax::transform::LegalizeOps(tvm::NullOpt),
  //     tvm::tir::transform::LowerTVMBuiltin(),
  //     tvm::tir::transform::LowerIntrin(),
  //     tvm::tir::transform::LowerThreadAllreduce(),
  //     tvm::tir::transform::LowerDeviceStorageAccessInfo(),
  //     tvm::tir::transform::LowerCustomDatatypes(),
  // });
  // irmodule = passseq(irmodule);
  // LOG_PRINT_VAR(irmodule);
  // auto vm = tvm::relax::transform::VMCodeLower(target, "executable")(irmodule);
  // tvm::Target target = tvm::Target("llvm");
  // LOG_PRINT_VAR(target->kind->name);
  // tvm::runtime::Module exe = tvm::codegen::Build(irmodule, target);
  // tvm::runtime::Module module = tvm::codegen::Build(irmodule, target);
  // LOG_PRINT_VAR(exe);
}

void Tirshift_rightTest() {}

void Tirlarge_uint_immTest() {}

void Tirq_multiply_shiftTest() {}

void Tiraddress_ofTest() {}

void Tirif_then_elseTest() {}

void TirisnullptrTest() {}

void TirisnanTest() {}

void TirpopcountTest() {}

void TirfmaTest() {}

void Tircall_externTest() {}

void Tircall_pure_externTest() {}

void Tircall_llvm_intrinTest() {}

void Tircall_llvm_pure_intrinTest() {}

void Tircall_spirv_pure_glsl450Test() {}

void TirprefetchTest() {}

void Tirtvm_access_ptrTest() {}

void Tirtvm_static_handleTest() {}

void Tirtvm_context_idTest() {}

void Tirtvm_tupleTest() {}

void Tirtvm_struct_getTest() {}

void Tirtvm_struct_setTest() {}

void Tirlookup_paramTest() {}

void Tirtvm_throw_last_errorTest() {}

void Tirtvm_stack_allocaTest() {}

void Tirtvm_stack_make_shapeTest() {}

void Tirtvm_stack_make_arrayTest() {}

void Tirtvm_call_packedTest() {}

void Tirtvm_call_cpackedTest() {}

void Tirtvm_call_trace_packedTest() {}

void Tirtvm_check_returnTest() {}

void Tirtvm_thread_contextTest() {}

void Tirtvm_thread_invariantTest() {}

void Tirtvm_call_packed_loweredTest() {}

void Tirtvm_call_cpacked_loweredTest() {}

void Tirtvm_call_trace_packed_loweredTest() {}

void Tirtvm_storage_syncTest() {}

void Tirtvm_warp_shuffleTest() {}

void Tirtvm_warp_shuffle_upTest() {}

void Tirtvm_warp_shuffle_downTest() {}

void Tirtvm_warp_activemaskTest() {}

void Tirtvm_global_barrier_kinitTest() {}

void Tirtvm_thread_allreduceTest() {}

void Tirtvm_load_matrix_syncTest() {}

void Tirtvm_mma_syncTest() {}

void Tirtvm_bmma_syncTest() {}

void Tirtvm_fill_fragmentTest() {}

void Tirtvm_store_matrix_syncTest() {}

void Tirptx_mmaTest() {}

void Tirptx_ldg32Test() {}

void Tirptx_mma_spTest() {}

void Tirptx_ldmatrixTest() {}

void Tirptx_cp_asyncTest() {}

void Tirptx_cp_async_bulkTest() {}

void Tirptx_commit_groupTest() {}

void Tirptx_wait_groupTest() {}

void Tirptx_cp_async_barrierTest() {}

void Tirptx_init_barrier_thread_countTest() {}

void Tirptx_arrive_barrierTest() {}

void Tirptx_arrive_barrier_expect_txTest() {}

void Tirptx_wait_barrierTest() {}

void Tircreate_barriersTest() {}

void Tirmma_storeTest() {}

void Tirmma_fillTest() {}

void Tirmake_filled_simdgroup_matrixTest() {}

void Tirsimdgroup_loadTest() {}

void Tirsimdgroup_storeTest() {}

void Tirsimdgroup_multiply_accumulateTest() {}

void TirvectorhighTest() {}

void TirvectorlowTest() {}

void TirvectorcombineTest() {}

void Tirdp4aTest() {}

void Tiratomic_addTest() {}

void Tirnd_mem_alloc_with_scopeTest() {}

void Tirtexture2d_storeTest() {}

void Tirtexture2d_loadTest() {}

void Tirdma_copyTest() {}

void Tirdma_waitTest() {}

void Tirdma_start_groupTest() {}

void Tirdma_end_groupTest() {}

void TirassumeTest() {}

void TirundefTest() {}

void Tirstart_profile_intrinsicTest() {}

void Tirend_profile_intrinsicTest() {}

void Tiranylist_getitemTest() {}

void Tiranylist_resetitemTest() {}

void Tiranylist_setitem_call_packedTest() {}

void Tiranylist_setitem_call_cpackedTest() {}

void TirvscaleTest() {}

void Tirget_active_lane_maskTest() {}

void Tirignore_loop_partitionTest() {}

}  // namespace builtin_test

// clang-format off
REGISTER_TEST_SUITE(builtin_test::TirretTest, tir_builtin_test_TirretTest);
REGISTER_TEST_SUITE(builtin_test::TirreinterpretTest, tir_builtin_test_TirreinterpretTest);
REGISTER_TEST_SUITE(builtin_test::TirlikelyTest, tir_builtin_test_TirlikelyTest);
REGISTER_TEST_SUITE(builtin_test::Tirbitwise_andTest, tir_builtin_test_Tirbitwise_andTest);
REGISTER_TEST_SUITE(builtin_test::Tirbitwise_orTest, tir_builtin_test_Tirbitwise_orTest);
REGISTER_TEST_SUITE(builtin_test::Tirbitwise_xorTest, tir_builtin_test_Tirbitwise_xorTest);
REGISTER_TEST_SUITE(builtin_test::Tirbitwise_notTest, tir_builtin_test_Tirbitwise_notTest);
REGISTER_TEST_SUITE(builtin_test::Tirshift_leftTest, tir_builtin_test_Tirshift_leftTest);
REGISTER_TEST_SUITE(builtin_test::Tirshift_rightTest, tir_builtin_test_Tirshift_rightTest);
REGISTER_TEST_SUITE(builtin_test::Tirlarge_uint_immTest, tir_builtin_test_Tirlarge_uint_immTest);
REGISTER_TEST_SUITE(builtin_test::Tirq_multiply_shiftTest, tir_builtin_test_Tirq_multiply_shiftTest);
REGISTER_TEST_SUITE(builtin_test::Tiraddress_ofTest, tir_builtin_test_Tiraddress_ofTest);
REGISTER_TEST_SUITE(builtin_test::Tirif_then_elseTest, tir_builtin_test_Tirif_then_elseTest);
REGISTER_TEST_SUITE(builtin_test::TirisnullptrTest, tir_builtin_test_TirisnullptrTest);
REGISTER_TEST_SUITE(builtin_test::TirisnanTest, tir_builtin_test_TirisnanTest);
REGISTER_TEST_SUITE(builtin_test::TirpopcountTest, tir_builtin_test_TirpopcountTest);
REGISTER_TEST_SUITE(builtin_test::TirfmaTest, tir_builtin_test_TirfmaTest);
REGISTER_TEST_SUITE(builtin_test::Tircall_externTest, tir_builtin_test_Tircall_externTest);
REGISTER_TEST_SUITE(builtin_test::Tircall_pure_externTest, tir_builtin_test_Tircall_pure_externTest);
REGISTER_TEST_SUITE(builtin_test::Tircall_llvm_intrinTest, tir_builtin_test_Tircall_llvm_intrinTest);
REGISTER_TEST_SUITE(builtin_test::Tircall_llvm_pure_intrinTest, tir_builtin_test_Tircall_llvm_pure_intrinTest);
REGISTER_TEST_SUITE(builtin_test::Tircall_spirv_pure_glsl450Test, tir_builtin_test_Tircall_spirv_pure_glsl450Test);
REGISTER_TEST_SUITE(builtin_test::TirprefetchTest, tir_builtin_test_TirprefetchTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_access_ptrTest, tir_builtin_test_Tirtvm_access_ptrTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_static_handleTest, tir_builtin_test_Tirtvm_static_handleTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_context_idTest, tir_builtin_test_Tirtvm_context_idTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_tupleTest, tir_builtin_test_Tirtvm_tupleTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_struct_getTest, tir_builtin_test_Tirtvm_struct_getTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_struct_setTest, tir_builtin_test_Tirtvm_struct_setTest);
REGISTER_TEST_SUITE(builtin_test::Tirlookup_paramTest, tir_builtin_test_Tirlookup_paramTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_throw_last_errorTest, tir_builtin_test_Tirtvm_throw_last_errorTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_stack_allocaTest, tir_builtin_test_Tirtvm_stack_allocaTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_stack_make_shapeTest, tir_builtin_test_Tirtvm_stack_make_shapeTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_stack_make_arrayTest, tir_builtin_test_Tirtvm_stack_make_arrayTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_call_packedTest, tir_builtin_test_Tirtvm_call_packedTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_call_cpackedTest, tir_builtin_test_Tirtvm_call_cpackedTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_call_trace_packedTest, tir_builtin_test_Tirtvm_call_trace_packedTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_check_returnTest, tir_builtin_test_Tirtvm_check_returnTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_thread_contextTest, tir_builtin_test_Tirtvm_thread_contextTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_thread_invariantTest, tir_builtin_test_Tirtvm_thread_invariantTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_call_packed_loweredTest, tir_builtin_test_Tirtvm_call_packed_loweredTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_call_cpacked_loweredTest, tir_builtin_test_Tirtvm_call_cpacked_loweredTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_call_trace_packed_loweredTest, tir_builtin_test_Tirtvm_call_trace_packed_loweredTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_storage_syncTest, tir_builtin_test_Tirtvm_storage_syncTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_warp_shuffleTest, tir_builtin_test_Tirtvm_warp_shuffleTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_warp_shuffle_upTest, tir_builtin_test_Tirtvm_warp_shuffle_upTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_warp_shuffle_downTest, tir_builtin_test_Tirtvm_warp_shuffle_downTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_warp_activemaskTest, tir_builtin_test_Tirtvm_warp_activemaskTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_global_barrier_kinitTest, tir_builtin_test_Tirtvm_global_barrier_kinitTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_thread_allreduceTest, tir_builtin_test_Tirtvm_thread_allreduceTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_load_matrix_syncTest, tir_builtin_test_Tirtvm_load_matrix_syncTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_mma_syncTest, tir_builtin_test_Tirtvm_mma_syncTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_bmma_syncTest, tir_builtin_test_Tirtvm_bmma_syncTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_fill_fragmentTest, tir_builtin_test_Tirtvm_fill_fragmentTest);
REGISTER_TEST_SUITE(builtin_test::Tirtvm_store_matrix_syncTest, tir_builtin_test_Tirtvm_store_matrix_syncTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_mmaTest, tir_builtin_test_Tirptx_mmaTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_ldg32Test, tir_builtin_test_Tirptx_ldg32Test);
REGISTER_TEST_SUITE(builtin_test::Tirptx_mma_spTest, tir_builtin_test_Tirptx_mma_spTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_ldmatrixTest, tir_builtin_test_Tirptx_ldmatrixTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_cp_asyncTest, tir_builtin_test_Tirptx_cp_asyncTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_cp_async_bulkTest, tir_builtin_test_Tirptx_cp_async_bulkTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_commit_groupTest, tir_builtin_test_Tirptx_commit_groupTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_wait_groupTest, tir_builtin_test_Tirptx_wait_groupTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_cp_async_barrierTest, tir_builtin_test_Tirptx_cp_async_barrierTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_init_barrier_thread_countTest, tir_builtin_test_Tirptx_init_barrier_thread_countTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_arrive_barrierTest, tir_builtin_test_Tirptx_arrive_barrierTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_arrive_barrier_expect_txTest, tir_builtin_test_Tirptx_arrive_barrier_expect_txTest);
REGISTER_TEST_SUITE(builtin_test::Tirptx_wait_barrierTest, tir_builtin_test_Tirptx_wait_barrierTest);
REGISTER_TEST_SUITE(builtin_test::Tircreate_barriersTest, tir_builtin_test_Tircreate_barriersTest);
REGISTER_TEST_SUITE(builtin_test::Tirmma_storeTest, tir_builtin_test_Tirmma_storeTest);
REGISTER_TEST_SUITE(builtin_test::Tirmma_fillTest, tir_builtin_test_Tirmma_fillTest);
REGISTER_TEST_SUITE(builtin_test::Tirmake_filled_simdgroup_matrixTest, tir_builtin_test_Tirmake_filled_simdgroup_matrixTest);
REGISTER_TEST_SUITE(builtin_test::Tirsimdgroup_loadTest, tir_builtin_test_Tirsimdgroup_loadTest);
REGISTER_TEST_SUITE(builtin_test::Tirsimdgroup_storeTest, tir_builtin_test_Tirsimdgroup_storeTest);
REGISTER_TEST_SUITE(builtin_test::Tirsimdgroup_multiply_accumulateTest, tir_builtin_test_Tirsimdgroup_multiply_accumulateTest);
REGISTER_TEST_SUITE(builtin_test::TirvectorhighTest, tir_builtin_test_TirvectorhighTest);
REGISTER_TEST_SUITE(builtin_test::TirvectorlowTest, tir_builtin_test_TirvectorlowTest);
REGISTER_TEST_SUITE(builtin_test::TirvectorcombineTest, tir_builtin_test_TirvectorcombineTest);
REGISTER_TEST_SUITE(builtin_test::Tirdp4aTest, tir_builtin_test_Tirdp4aTest);
REGISTER_TEST_SUITE(builtin_test::Tiratomic_addTest, tir_builtin_test_Tiratomic_addTest);
REGISTER_TEST_SUITE(builtin_test::Tirnd_mem_alloc_with_scopeTest, tir_builtin_test_Tirnd_mem_alloc_with_scopeTest);
REGISTER_TEST_SUITE(builtin_test::Tirtexture2d_storeTest, tir_builtin_test_Tirtexture2d_storeTest);
REGISTER_TEST_SUITE(builtin_test::Tirtexture2d_loadTest, tir_builtin_test_Tirtexture2d_loadTest);
REGISTER_TEST_SUITE(builtin_test::Tirdma_copyTest, tir_builtin_test_Tirdma_copyTest);
REGISTER_TEST_SUITE(builtin_test::Tirdma_waitTest, tir_builtin_test_Tirdma_waitTest);
REGISTER_TEST_SUITE(builtin_test::Tirdma_start_groupTest, tir_builtin_test_Tirdma_start_groupTest);
REGISTER_TEST_SUITE(builtin_test::Tirdma_end_groupTest, tir_builtin_test_Tirdma_end_groupTest);
REGISTER_TEST_SUITE(builtin_test::TirassumeTest, tir_builtin_test_TirassumeTest);
REGISTER_TEST_SUITE(builtin_test::TirundefTest, tir_builtin_test_TirundefTest);
REGISTER_TEST_SUITE(builtin_test::Tirstart_profile_intrinsicTest, tir_builtin_test_Tirstart_profile_intrinsicTest);
REGISTER_TEST_SUITE(builtin_test::Tirend_profile_intrinsicTest, tir_builtin_test_Tirend_profile_intrinsicTest);
REGISTER_TEST_SUITE(builtin_test::Tiranylist_getitemTest, tir_builtin_test_Tiranylist_getitemTest);
REGISTER_TEST_SUITE(builtin_test::Tiranylist_resetitemTest, tir_builtin_test_Tiranylist_resetitemTest);
REGISTER_TEST_SUITE(builtin_test::Tiranylist_setitem_call_packedTest, tir_builtin_test_Tiranylist_setitem_call_packedTest);
REGISTER_TEST_SUITE(builtin_test::Tiranylist_setitem_call_cpackedTest, tir_builtin_test_Tiranylist_setitem_call_cpackedTest);
REGISTER_TEST_SUITE(builtin_test::TirvscaleTest, tir_builtin_test_TirvscaleTest);
REGISTER_TEST_SUITE(builtin_test::Tirget_active_lane_maskTest, tir_builtin_test_Tirget_active_lane_maskTest);
REGISTER_TEST_SUITE(builtin_test::Tirignore_loop_partitionTest, tir_builtin_test_Tirignore_loop_partitionTest);
// clang-format on
