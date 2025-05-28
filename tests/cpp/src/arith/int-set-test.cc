#include "arith/int-set-test.h"
#include "test-func-registry.h"

namespace int_set_test {

void ArithIntSetTest() {
  LOG_SPLIT_LINE("ArithIntSetTest");

  Var vec{"vec", tvm::DataType::Int(32, 4)};
  IntSet intsetvec = IntSet::Vector(vec);
  LOG_PRINT_VAR(intsetvec.min());
  LOG_PRINT_VAR(intsetvec.max());
  LOG_PRINT_VAR(intsetvec.MatchRange({1, 3}));

  IntSet intsetsinglepoint = IntSet::SinglePoint(1.2f);
  LOG_PRINT_VAR(intsetsinglepoint.min());
  LOG_PRINT_VAR(intsetsinglepoint.max());
  LOG_PRINT_VAR(intsetsinglepoint.MatchRange({1, 3}));
  LOG_PRINT_VAR(intsetsinglepoint.PointValue());

  IntSet intsetfromminextent = IntSet::FromMinExtent(1, 3);
  LOG_PRINT_VAR(intsetfromminextent.min());
  LOG_PRINT_VAR(intsetfromminextent.max());
  LOG_PRINT_VAR(intsetfromminextent.MatchRange({1, 3}));

  // Range[1, 3).
  IntSet intsetfromrange = IntSet::FromRange({1, 3});
  LOG_PRINT_VAR(intsetfromrange.min());
  LOG_PRINT_VAR(intsetfromrange.max());
  LOG_PRINT_VAR(intsetfromrange.MatchRange({1, 3}));

  IntSet intsetinterval = IntSet::Interval(1, 3);
  LOG_PRINT_VAR(intsetinterval.min());
  LOG_PRINT_VAR(intsetinterval.max());
  LOG_PRINT_VAR(intsetinterval.MatchRange({1, 3}));

  IntSet intseteverything = IntSet::Everything();
  LOG_PRINT_VAR(intseteverything.min());
  LOG_PRINT_VAR(intseteverything.max());
  LOG_PRINT_VAR(intseteverything.MatchRange({1, 3}));

  IntSet intsetnothing = IntSet::Nothing();
  LOG_PRINT_VAR(intsetnothing.min());
  LOG_PRINT_VAR(intsetnothing.max());
  LOG_PRINT_VAR(intsetnothing.MatchRange({1, 3}));
}

}  // namespace int_set_test

REGISTER_TEST_SUITE(int_set_test::ArithIntSetTest, arith_int_set_test_ArithIntSetTest);
