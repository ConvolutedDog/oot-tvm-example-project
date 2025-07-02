#include "meta_schedule/per_store_feature_test.h"
#include <cstdint>

namespace per_store_feature_test {
void RelaxAndUnionTest() {
  /// A vector of index expressions representing different memory access.
  std::vector<MultiIndex> multi_indices = {
      {0, 1},
      {1, 2},
      {3, 3}
  };
  int64_t numel = 0;
  tvm::arith::Analyzer analyzer;
  /// @note
  /// `RelaxAndUnion`将会计算出每一个维度覆盖的范围，例如：第一个维度有0,1,3,因此覆盖了0~3
  /// 第二个维度有1,2,3,因此覆盖了1~3
  IntVec access_shape = RelaxAndUnion(multi_indices, &numel, &analyzer);
  LOG_PRINT_VAR(access_shape.size());
  LOG_PRINT_VAR(access_shape[0]);  /// 1st dimension covers 4 elements(0,1,2,3)
  LOG_PRINT_VAR(access_shape[1]);  /// 2nd dimension covers 3 elements(1,2,3)
  LOG_PRINT_VAR(numel);            /// 4 * 3 = 12

  std::vector<MultiIndex> multi_indices2 = {
      {1, 1, 3},
      {3, 9, 7},
      {6, 8, 4}
  };
  int64_t numel2 = 0;
  IntVec access_shape2 = RelaxAndUnion(multi_indices2, &numel2, &analyzer);
  LOG_PRINT_VAR(access_shape2.size());
  LOG_PRINT_VAR(access_shape2[0]);  /// 1st dimension covers 7 elements(0,1,2,3,4,5,6)
  LOG_PRINT_VAR(access_shape2[1]);  /// 2nd dimension covers 9 elements(1~9)
  LOG_PRINT_VAR(access_shape2[2]);  /// 3rd dimension covers 5 elements(3,4,5,6,7)
  LOG_PRINT_VAR(numel2);            /// 7 * 9 * 5 = 315
}

void GetVarStrideTest() {
  Var i("i"), j("j");

  std::vector<MultiIndex> multi_indices = {
      {4 * i, 1 + 3 * j},
      {2 * i, 2 * j    },
      {3 * i, 3 * j    }
  };
  IntVec buffer_stride = {10, 1};

  /// @note Computes the minimal stride of a variable in several multi-index patterns.
  /// @example For a buffer with `buffer_stride = {10, 1}` (row major, where traversing a
  /// row skips 10 elements, and a column skips 1 element):
  /// - If the multi-index pattern for `i` is `{4 * i, 1 + 3 * j}`, then the stride of `i`
  /// is ​**40**, because the offset between `4*i` and `4*(i+1)` is `4 * 10 = 40`
  /// elements.
  /// - If the multi-index pattern for `j` is `{2 * i, 2 * j}`, then the stride of `j` is
  /// ​**2**, since the offset between `2*j` and `2*(j+1)` along columns is `2 * 1 = 2`
  /// elements.

  int64_t stride_i = GetVarStride(multi_indices, buffer_stride, i);
  int64_t stride_j = GetVarStride(multi_indices, buffer_stride, j);
  LOG_PRINT_VAR(stride_i);
  LOG_PRINT_VAR(stride_j);

  std::vector<MultiIndex> multi_indices2 = {
      {2 * i + 3 * j, 1 + 3 * j},
      {2 * i + j,     4 * j    },
      {3 * i,         3 * j    }
  };
  IntVec buffer_stride2 = {20, 10};
  int64_t stride_i2 = GetVarStride(multi_indices2, buffer_stride2, i);
  int64_t stride_j2 = GetVarStride(multi_indices2, buffer_stride2, j);
  LOG_PRINT_VAR(stride_i2);
  LOG_PRINT_VAR(stride_j2);
}
}  // namespace per_store_feature_test

REGISTER_TEST_SUITE(per_store_feature_test::RelaxAndUnionTest,
                    meta_schedule_RelaxAndUnionTest);
REGISTER_TEST_SUITE(per_store_feature_test::GetVarStrideTest,
                    meta_schedule_GetVarStrideTest);
