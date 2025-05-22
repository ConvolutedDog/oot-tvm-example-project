#include "tvm/tir/builtin.h"

namespace builtin_test {

using tvm::tir::builtin::address_of;
using tvm::tir::builtin::anylist_getitem;
using tvm::tir::builtin::anylist_resetitem;
using tvm::tir::builtin::anylist_setitem_call_cpacked;
using tvm::tir::builtin::anylist_setitem_call_packed;
using tvm::tir::builtin::assume;
using tvm::tir::builtin::atomic_add;
using tvm::tir::builtin::bitwise_and;
using tvm::tir::builtin::bitwise_not;
using tvm::tir::builtin::bitwise_or;
using tvm::tir::builtin::bitwise_xor;
using tvm::tir::builtin::call_extern;
using tvm::tir::builtin::call_llvm_intrin;
using tvm::tir::builtin::call_llvm_pure_intrin;
using tvm::tir::builtin::call_pure_extern;
using tvm::tir::builtin::call_spirv_pure_glsl450;
using tvm::tir::builtin::create_barriers;
using tvm::tir::builtin::dma_copy;
using tvm::tir::builtin::dma_end_group;
using tvm::tir::builtin::dma_start_group;
using tvm::tir::builtin::dma_wait;
using tvm::tir::builtin::dp4a;
using tvm::tir::builtin::end_profile_intrinsic;
using tvm::tir::builtin::fma;
using tvm::tir::builtin::get_active_lane_mask;
using tvm::tir::builtin::if_then_else;
using tvm::tir::builtin::ignore_loop_partition;
using tvm::tir::builtin::isnan;
using tvm::tir::builtin::isnullptr;
using tvm::tir::builtin::large_uint_imm;
using tvm::tir::builtin::likely;
using tvm::tir::builtin::lookup_param;
using tvm::tir::builtin::make_filled_simdgroup_matrix;
using tvm::tir::builtin::mma_fill;
using tvm::tir::builtin::mma_store;
using tvm::tir::builtin::nd_mem_alloc_with_scope;
using tvm::tir::builtin::popcount;
using tvm::tir::builtin::prefetch;
using tvm::tir::builtin::ptx_arrive_barrier;
using tvm::tir::builtin::ptx_arrive_barrier_expect_tx;
using tvm::tir::builtin::ptx_commit_group;
using tvm::tir::builtin::ptx_cp_async;
using tvm::tir::builtin::ptx_cp_async_barrier;
using tvm::tir::builtin::ptx_cp_async_bulk;
using tvm::tir::builtin::ptx_init_barrier_thread_count;
using tvm::tir::builtin::ptx_ldg32;
using tvm::tir::builtin::ptx_ldmatrix;
using tvm::tir::builtin::ptx_mma;
using tvm::tir::builtin::ptx_mma_sp;
using tvm::tir::builtin::ptx_wait_barrier;
using tvm::tir::builtin::ptx_wait_group;
using tvm::tir::builtin::q_multiply_shift;
using tvm::tir::builtin::reinterpret;
using tvm::tir::builtin::ret;
using tvm::tir::builtin::shift_left;
using tvm::tir::builtin::shift_right;
using tvm::tir::builtin::simdgroup_load;
using tvm::tir::builtin::simdgroup_multiply_accumulate;
using tvm::tir::builtin::simdgroup_store;
using tvm::tir::builtin::start_profile_intrinsic;
using tvm::tir::builtin::texture2d_load;
using tvm::tir::builtin::texture2d_store;
using tvm::tir::builtin::tvm_access_ptr;
using tvm::tir::builtin::tvm_bmma_sync;
using tvm::tir::builtin::tvm_call_cpacked;
using tvm::tir::builtin::tvm_call_cpacked_lowered;
using tvm::tir::builtin::tvm_call_packed;
using tvm::tir::builtin::tvm_call_packed_lowered;
using tvm::tir::builtin::tvm_call_trace_packed;
using tvm::tir::builtin::tvm_call_trace_packed_lowered;
using tvm::tir::builtin::tvm_check_return;
using tvm::tir::builtin::tvm_context_id;
using tvm::tir::builtin::tvm_fill_fragment;
using tvm::tir::builtin::tvm_global_barrier_kinit;
using tvm::tir::builtin::tvm_load_matrix_sync;
using tvm::tir::builtin::tvm_mma_sync;
using tvm::tir::builtin::tvm_stack_alloca;
using tvm::tir::builtin::tvm_stack_make_array;
using tvm::tir::builtin::tvm_stack_make_shape;
using tvm::tir::builtin::tvm_static_handle;
using tvm::tir::builtin::tvm_storage_sync;
using tvm::tir::builtin::tvm_store_matrix_sync;
using tvm::tir::builtin::tvm_struct_get;
using tvm::tir::builtin::tvm_struct_set;
using tvm::tir::builtin::tvm_thread_allreduce;
using tvm::tir::builtin::tvm_thread_context;
using tvm::tir::builtin::tvm_thread_invariant;
using tvm::tir::builtin::tvm_throw_last_error;
using tvm::tir::builtin::tvm_tuple;
using tvm::tir::builtin::tvm_warp_activemask;
using tvm::tir::builtin::tvm_warp_shuffle;
using tvm::tir::builtin::tvm_warp_shuffle_down;
using tvm::tir::builtin::tvm_warp_shuffle_up;
using tvm::tir::builtin::undef;
using tvm::tir::builtin::vectorcombine;
using tvm::tir::builtin::vectorhigh;
using tvm::tir::builtin::vectorlow;
using tvm::tir::builtin::vscale;

void TirretTest();                            // NOLINT
void TirreinterpretTest();                    // NOLINT
void TirlikelyTest();                         // NOLINT
void Tirbitwise_andTest();                    // NOLINT
void Tirbitwise_orTest();                     // NOLINT
void Tirbitwise_xorTest();                    // NOLINT
void Tirbitwise_notTest();                    // NOLINT
void Tirshift_leftTest();                     // NOLINT
void Tirshift_rightTest();                    // NOLINT
void Tirlarge_uint_immTest();                 // NOLINT
void Tirq_multiply_shiftTest();               // NOLINT
void Tiraddress_ofTest();                     // NOLINT
void Tirif_then_elseTest();                   // NOLINT
void TirisnullptrTest();                      // NOLINT
void TirisnanTest();                          // NOLINT
void TirpopcountTest();                       // NOLINT
void TirfmaTest();                            // NOLINT
void Tircall_externTest();                    // NOLINT
void Tircall_pure_externTest();               // NOLINT
void Tircall_llvm_intrinTest();               // NOLINT
void Tircall_llvm_pure_intrinTest();          // NOLINT
void Tircall_spirv_pure_glsl450Test();        // NOLINT
void TirprefetchTest();                       // NOLINT
void Tirtvm_access_ptrTest();                 // NOLINT
void Tirtvm_static_handleTest();              // NOLINT
void Tirtvm_context_idTest();                 // NOLINT
void Tirtvm_tupleTest();                      // NOLINT
void Tirtvm_struct_getTest();                 // NOLINT
void Tirtvm_struct_setTest();                 // NOLINT
void Tirlookup_paramTest();                   // NOLINT
void Tirtvm_throw_last_errorTest();           // NOLINT
void Tirtvm_stack_allocaTest();               // NOLINT
void Tirtvm_stack_make_shapeTest();           // NOLINT
void Tirtvm_stack_make_arrayTest();           // NOLINT
void Tirtvm_call_packedTest();                // NOLINT
void Tirtvm_call_cpackedTest();               // NOLINT
void Tirtvm_call_trace_packedTest();          // NOLINT
void Tirtvm_check_returnTest();               // NOLINT
void Tirtvm_thread_contextTest();             // NOLINT
void Tirtvm_thread_invariantTest();           // NOLINT
void Tirtvm_call_packed_loweredTest();        // NOLINT
void Tirtvm_call_cpacked_loweredTest();       // NOLINT
void Tirtvm_call_trace_packed_loweredTest();  // NOLINT
void Tirtvm_storage_syncTest();               // NOLINT
void Tirtvm_warp_shuffleTest();               // NOLINT
void Tirtvm_warp_shuffle_upTest();            // NOLINT
void Tirtvm_warp_shuffle_downTest();          // NOLINT
void Tirtvm_warp_activemaskTest();            // NOLINT
void Tirtvm_global_barrier_kinitTest();       // NOLINT
void Tirtvm_thread_allreduceTest();           // NOLINT
void Tirtvm_load_matrix_syncTest();           // NOLINT
void Tirtvm_mma_syncTest();                   // NOLINT
void Tirtvm_bmma_syncTest();                  // NOLINT
void Tirtvm_fill_fragmentTest();              // NOLINT
void Tirtvm_store_matrix_syncTest();          // NOLINT
void Tirptx_mmaTest();                        // NOLINT
void Tirptx_ldg32Test();                      // NOLINT
void Tirptx_mma_spTest();                     // NOLINT
void Tirptx_ldmatrixTest();                   // NOLINT
void Tirptx_cp_asyncTest();                   // NOLINT
void Tirptx_cp_async_bulkTest();              // NOLINT
void Tirptx_commit_groupTest();               // NOLINT
void Tirptx_wait_groupTest();                 // NOLINT
void Tirptx_cp_async_barrierTest();           // NOLINT
void Tirptx_init_barrier_thread_countTest();  // NOLINT
void Tirptx_arrive_barrierTest();             // NOLINT
void Tirptx_arrive_barrier_expect_txTest();   // NOLINT
void Tirptx_wait_barrierTest();               // NOLINT
void Tircreate_barriersTest();                // NOLINT
void Tirmma_storeTest();                      // NOLINT
void Tirmma_fillTest();                       // NOLINT
void Tirmake_filled_simdgroup_matrixTest();   // NOLINT
void Tirsimdgroup_loadTest();                 // NOLINT
void Tirsimdgroup_storeTest();                // NOLINT
void Tirsimdgroup_multiply_accumulateTest();  // NOLINT
void TirvectorhighTest();                     // NOLINT
void TirvectorlowTest();                      // NOLINT
void TirvectorcombineTest();                  // NOLINT
void Tirdp4aTest();                           // NOLINT
void Tiratomic_addTest();                     // NOLINT
void Tirnd_mem_alloc_with_scopeTest();        // NOLINT
void Tirtexture2d_storeTest();                // NOLINT
void Tirtexture2d_loadTest();                 // NOLINT
void Tirdma_copyTest();                       // NOLINT
void Tirdma_waitTest();                       // NOLINT
void Tirdma_start_groupTest();                // NOLINT
void Tirdma_end_groupTest();                  // NOLINT
void TirassumeTest();                         // NOLINT
void TirundefTest();                          // NOLINT
void Tirstart_profile_intrinsicTest();        // NOLINT
void Tirend_profile_intrinsicTest();          // NOLINT
void Tiranylist_getitemTest();                // NOLINT
void Tiranylist_resetitemTest();              // NOLINT
void Tiranylist_setitem_call_packedTest();    // NOLINT
void Tiranylist_setitem_call_cpackedTest();   // NOLINT
void TirvscaleTest();                         // NOLINT
void Tirget_active_lane_maskTest();           // NOLINT
void Tirignore_loop_partitionTest();          // NOLINT

}  // namespace builtin_test
