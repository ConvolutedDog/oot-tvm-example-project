#include "tir/data-layout-test.h"
#include "test-func-registry.h"
#include <cstddef>
#include <tvm/runtime/logging.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/var.h>

namespace data_layout_test {

void TirLayoutAxisTest() {
  LOG_SPLIT_LINE("TirLayoutAxisTest");

  const LayoutAxis &axisbyname = LayoutAxis::Get('x');
  LOG_PRINT_VAR(axisbyname);  // x

  const LayoutAxis &axisbystring = LayoutAxis::Get("w");
  LOG_PRINT_VAR(axisbystring);  // w

  IterVar itervar{
      Range{0, 4},
      Var{"P"},
      IterVarType::kDataPar
  };
  const LayoutAxis &axisbyitervar = LayoutAxis::Get(itervar);
  LOG_PRINT_VAR(axisbyitervar);  // P
  /// Primal: 'A' <= name <= 'Z'.
  LOG_PRINT_VAR(axisbyitervar.IsPrimal());  // 1
  LOG_PRINT_VAR(axisbyitervar.name());      //  P
  /// ToPrimal(): Lowercase -> Uppercase, Uppercase -> Uppercae.
  LOG_PRINT_VAR(axisbyitervar.ToPrimal());  // P
  /// ToDual(): Lowercase -> Uppercase, Uppercase -> Lowercase.
  LOG_PRINT_VAR(axisbyitervar.ToDual());  // p
  /// ToSubordinate(): Lowercase -> Lowercase, Uppercase -> Lowercase.
  LOG_PRINT_VAR(axisbyitervar.ToSubordinate());  // p
  LOG_PRINT_VAR(axisbyitervar == axisbyname);    // 0
}

/// @brief Layout is to describe how data is organized within an N-dimention tensor. It is
/// composed of upper cases, lower cases and numbers, where upper case indicates a primal
/// axis and the corresponding lower case with factor size indicates the subordinate axis.
/// For example, NCHW16c can describe a 5-D tensor of [batch_size, channel, height, width,
/// channel_block]. Here subordinate axis channel_block=16 is the factor size of the
/// primal axis C (channel). Layout for scalar is defined, while both its name and axes
/// have size 0.
void TirLayoutTest() {
  LOG_SPLIT_LINE("TirLayoutTest");

  // clang-format off
  IterVar N{{0, 1}, Var{"N"}, IterVarType::kDataPar}; // NOLINT
  IterVar C{{0, 1}, Var{"C"}, IterVarType::kDataPar}; // NOLINT
  IterVar H{{0, 1}, Var{"H"}, IterVarType::kDataPar}; // NOLINT
  IterVar W{{0, 1}, Var{"W"}, IterVarType::kDataPar}; // NOLINT
  IterVar c{{0, 16}, Var{"c"}, IterVarType::kDataPar};
  // clang-format on

  LOG_SPLIT_LINE("Construct with String");
  Layout layout{"NCHW16c"};
  LOG_PRINT_VAR(layout);  // NCHW16c
  LOG_PRINT_VAR(layout->axes);
  /// Output:
  ///   [T.iter_var(N, T.Range(0, N_shape), "DataPar", ""),
  ///    T.iter_var(C, T.Range(0, C_shape), "DataPar", ""),
  ///    T.iter_var(H, T.Range(0, H_shape), "DataPar", ""),
  ///    T.iter_var(W, T.Range(0, W_shape), "DataPar", ""),
  ///    T.iter_var(c, T.Range(0, 16), "DataPar", "")]
  LOG_PRINT_VAR(layout.ndim());         // 5
  LOG_PRINT_VAR(layout.ndim_primal());  // 4
  LOG_PRINT_VAR(layout.name());         // NCHW16c

  LOG_SPLIT_LINE("Construct with IterVars");
  Layout layoutNCHWc = Layout({N, C, H, W, c});
  LOG_PRINT_VAR(layoutNCHWc);  // 1N1C1H1W16c
  LOG_PRINT_VAR(layoutNCHWc->axes);
  /// Output:
  ///   [T.iter_var(N, T.Range(0, 1), "DataPar", ""),
  ///    T.iter_var(C, T.Range(0, 1), "DataPar", ""),
  ///    T.iter_var(H, T.Range(0, 1), "DataPar", ""),
  ///    T.iter_var(W, T.Range(0, 1), "DataPar", ""),
  ///    T.iter_var(c, T.Range(0, 16), "DataPar", "")]
  LOG_PRINT_VAR(layoutNCHWc.ndim());         // 5
  LOG_PRINT_VAR(layoutNCHWc.ndim_primal());  // 4

  LOG_SPLIT_LINE("SubLayout()");
  Layout sublayout = layoutNCHWc.SubLayout(0, 3);
  LOG_PRINT_VAR(sublayout);  // 1N1C1H
  LOG_PRINT_VAR(sublayout->axes);
  /// Output:
  ///   [T.iter_var(N, T.Range(0, 1), "DataPar", ""),
  ///    T.iter_var(C, T.Range(0, 1), "DataPar", ""),
  ///    T.iter_var(H, T.Range(0, 1), "DataPar", "")]
  LOG_PRINT_VAR(sublayout.ndim());         // 3
  LOG_PRINT_VAR(sublayout.ndim_primal());  // 3

  LOG_SPLIT_LINE("Split()");
  const LayoutAxis &newaxis = LayoutAxis::Get('H');
  Layout splittedlayout = layoutNCHWc.Split(newaxis, 1, 16);
  LOG_PRINT_VAR(splittedlayout);  // 1N16h1C1H1W16c
  LOG_PRINT_VAR(splittedlayout->axes);
  /// Output:
  ///  [T.iter_var(N, T.Range(0, 1), "DataPar", ""),
  ///   T.iter_var(h, T.Range(0, 16), "DataPar", ""),
  ///   T.iter_var(C, T.Range(0, 1), "DataPar", ""),
  ///   T.iter_var(H, T.Range(0, 1), "DataPar", ""),
  ///   T.iter_var(W, T.Range(0, 1), "DataPar", ""),
  ///   T.iter_var(c, T.Range(0, 16), "DataPar", "")]
  LOG_PRINT_VAR(splittedlayout.ndim());         // 6
  LOG_PRINT_VAR(splittedlayout.ndim_primal());  // 4

  LOG_SPLIT_LINE("Constructor with String");
  Layout layoutname{"N"};
  LOG_PRINT_VAR(layoutname);  // N
  LOG_PRINT_VAR(layoutname->axes);
  /// Output:
  ///   [T.iter_var(N, T.Range(0, N_shape), "DataPar", "")]
  LOG_PRINT_VAR(layoutname.ndim());         // 1
  LOG_PRINT_VAR(layoutname.ndim_primal());  // 1

  LOG_SPLIT_LINE("Undef()");
  Layout undef = Layout::Undef();
  LOG_PRINT_VAR(undef);                // __undef__
  LOG_PRINT_VAR(undef.ndim());         // 0
  LOG_PRINT_VAR(undef.ndim_primal());  // 0

  LOG_SPLIT_LINE("ExpandPrimal()");
  /// ExpandPrimal(): Returns a new layout where the dims have been expanded to match the
  /// primal dimensions. This method will seek Primal axes of `dst_layout`and insert them
  /// to the start of current layout.
  Layout expandedlayout = layout.ExpandPrimal(Layout{"PL16l8p"});
  LOG_PRINT_VAR(expandedlayout);  // PLNCHW16c
  LOG_PRINT_VAR(expandedlayout->axes);
  /// Output:
  ///   [T.iter_var(P, T.Range(0, P_shape), "DataPar", ""),
  ///    T.iter_var(L, T.Range(0, L_shape), "DataPar", ""),
  ///    T.iter_var(N, T.Range(0, N_shape), "DataPar", ""),
  ///    T.iter_var(C, T.Range(0, C_shape), "DataPar", ""),
  ///    T.iter_var(H, T.Range(0, H_shape), "DataPar", ""),
  ///    T.iter_var(W, T.Range(0, W_shape), "DataPar", ""),
  ///    T.iter_var(c, T.Range(0, 16), "DataPar", "")]
  LOG_PRINT_VAR(expandedlayout.ndim());         // 7
  LOG_PRINT_VAR(expandedlayout.ndim_primal());  // 6

  LOG_SPLIT_LINE("IndexOf()");
  const LayoutAxis &l = LayoutAxis::Get("c");
  LOG_PRINT_VAR(expandedlayout.IndexOf(l));  // 6

  LOG_SPLIT_LINE("Contains()");
  LOG_PRINT_VAR(expandedlayout.Contains(l));  // 1
  const LayoutAxis &l2 = LayoutAxis::Get("d");
  LOG_PRINT_VAR(expandedlayout.Contains(l2));  // 0

  LOG_SPLIT_LINE("operator[]");
  LOG_PRINT_VAR(expandedlayout[6]);  // c
  LOG_PRINT_VAR(expandedlayout[5]);  // W

  LOG_SPLIT_LINE("Equals()");
  LOG_PRINT_VAR(expandedlayout.Equals(expandedlayout));  // 1
  LOG_PRINT_VAR(layoutNCHWc.Equals(layout));             // 0
}

void TirBijectiveLayoutTest() {
  LOG_SPLIT_LINE("TirBijectiveLayoutTest");

  Layout src{"NCHW16c"};
  Layout dst{"NCHW"};
  /// @brief Bijective function mapping for data layout transformation. Given two Layout,
  /// BijectiveLayout build and store the mapping rules, provides API to transform
  /// N-dimention tensor from the source indices (i0, i1, .., im) to the destination
  /// indices (j0, j1, .., jm).
  BijectiveLayout bijectivelayout(src, dst);
  LOG_PRINT_VAR(bijectivelayout);
  LOG_PRINT_VAR(bijectivelayout->index_forward_rule);
  LOG_PRINT_VAR(bijectivelayout->index_backward_rule);
  LOG_PRINT_VAR(bijectivelayout->shape_forward_rule);
  LOG_PRINT_VAR(bijectivelayout->shape_backward_rule);

  /// Given the source shape, infer the destination shape.
  /// @note The last dim of `Shape` must be 16.
  LOG_PRINT_VAR(bijectivelayout.ForwardShape({1, 2, 3, 4, 16}));  // [1, 32, 3, 4]
                                                                  // 32 = 2 * 16

  /// Given the destination shape, recover the source shape.
  LOG_PRINT_VAR(bijectivelayout.BackwardShape({1, 32, 3, 4}));  // [1, 2, 3, 4, 16]
                                                                // 2 = (32 + 16 - 1) / 16

  LOG_PRINT_VAR(bijectivelayout.ForwardShape({1, 3, 4, 6, 16}));  // [1, 48, 4, 6]
  LOG_PRINT_VAR(bijectivelayout.BackwardShape({1, 50, 4, 6}));    // [1, 4, 4, 6, 16]

  /// Given the source indices, infer the destination indices.
  /// @note The last dim of `indices` must be less than or equals to 16.
  LOG_PRINT_VAR(bijectivelayout.ForwardIndex({0, 1, 2, 3, 16}));  // [0, 32, 2, 3]
                                                                  // 32 = 1 * 16 + 16

  /// Given the destination indices, recover the source indices.
  LOG_PRINT_VAR(
      bijectivelayout.BackwardIndex({1, 32, 2, 3}));  // [1, 2, 2, 3, 0]
                                                      // 2 = 32 / 16, 0 = 32 % 16
}

}  // namespace data_layout_test

REGISTER_TEST_SUITE(data_layout_test::TirLayoutAxisTest,
                    tir_data_layout_test_TirLayoutAxisTest);
REGISTER_TEST_SUITE(data_layout_test::TirLayoutTest, tir_data_layout_test_TirLayoutTest);
REGISTER_TEST_SUITE(data_layout_test::TirBijectiveLayoutTest,
                    tir_data_layout_test_TirBijectiveLayoutTest);
