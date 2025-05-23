#include "tir/index-map-test.h"
#include "test-func-registry.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/index_map.h>

namespace index_map_test {

/// @brief `IndexMap` defines a mapping between two representations of indices into a
/// buffer. This is primarily used for layout transformations of Buffer objects.
void TirIndexMapTest() {
  LOG_SPLIT_LINE("TirIndexMapTest");

  Var x("x", DataType::Int(32));
  Var y("y", DataType::Int(32));
  IndexMap im{
      {x,      y     },
      {y + x,  x - y },
      IndexMap{{x, y}, {y, x}}
  };
  LOG_PRINT_VAR(im);  // T.index_map(lambda x, y: (y + x, x - y))
  LOG_PRINT_VAR(im->inverse_index_map);

  Var a{"a"}, b{"b"}, c{"c"}, d{"d"};
  IndexMap im2{
      {a, b, c, d},
      {a * 4 + c, b * 16 + d}
  };
  LOG_PRINT_VAR(im2);  // T.index_map(lambda a, b, c, d: (a * 4 + c, b * 16 + d))
  LOG_PRINT_VAR(im2->initial_indices);    // [a, b, c, d]
  LOG_PRINT_VAR(im2->final_indices);      // [a * 4 + c, b * 16 + d]
  LOG_PRINT_VAR(im2->inverse_index_map);  // nullptr

  tvm::arith::Analyzer analyzer;
  LOG_PRINT_VAR(im2->MapIndices({1, 2, 3, 4}, &analyzer));  // [7, 36]
  LOG_PRINT_VAR(im2->MapRanges(
      {
          {1, 3}, // [1, 3)
          {3, 5}, // [3, 5)
          {5, 7}, // [5, 7)
          {7, 9}  // [7, 9)
  },
      &analyzer));  // [I.Range(9, 15), I.Range(55, 73)]
  /// `MapShape({1, 2, 3, 4})` <=> `MapRanges({{0, 1}, {0, 2}, {0, 3}, {0, 4}})`.
  LOG_PRINT_VAR(im2->MapShape({1, 2, 3, 4}, &analyzer));  // [3, 20]
  LOG_PRINT_VAR(im2->MapRanges(
      {
          {0, 1}, // [0, 1)
          {0, 2}, // [0, 2)
          {0, 3}, // [0, 3)
          {0, 4}  // [0, 4)
  },
      &analyzer));  // [I.Range(0, 3), I.Range(0, 20)]

  im2 = im2.RenameVariables([](const Var &var) { return var->name_hint + "_new"; });
  LOG_PRINT_VAR(im2);  // T.index_map(lambda a_new, b_new, c_new, d_new: (a_new * 4 +
                       // c_new, b_new * 16 + d_new))

  im2 = im2.FromFunc(4, [](const Array<Var> &var) {
    return Array<PrimExpr>{{var[0] * 4 + var[2]}, {var[1] * 16 + var[3]}};
  });
  LOG_PRINT_VAR(im2);  // T.index_map(lambda i0, i1, i2, i3: (i0 * 4 + i2, i1 * 16 + i3))

  /// @todo (yangjianchao) Supplement `NonSurjectiveInverse()` and `Inverse()`.
}

void TirSubstituteTest() {
  LOG_SPLIT_LINE("TirSubstituteTest");

  IndexMap im = IndexMap::FromFunc(4, [](const Array<Var> &var) {
    return Array<PrimExpr>{{var[0] * 4 + var[2]}, {var[1] * 16 + var[3]}};
  });
  LOG_PRINT_VAR(im);  // T.index_map(lambda i0, i1, i2, i3: (i0 * 4 + i2, i1 * 16 + i3))

  im = Substitute(im, [](const Var &var) { return PrimExpr{var + 1}; });
  LOG_PRINT_VAR(im);  // T.index_map(lambda i0, i1, i2, i3: ((i0 + 1) * 4 + (i2 + 1), (i1
                      // + 1) * 16 + (i3 + 1)))
}

}  // namespace index_map_test

REGISTER_TEST_SUITE(index_map_test::TirIndexMapTest, tir_index_map_test_TirIndexMapTest);
REGISTER_TEST_SUITE(index_map_test::TirSubstituteTest,
                    tir_index_map_test_TirSubstituteTest);
