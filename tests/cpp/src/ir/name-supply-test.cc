#include "ir/name-supply-test.h"
#include "test-func-registry.h"

namespace name_supply_test {

void IrNameSupplyTest() {
  LOG_SPLIT_LINE("IrNameSupplyTest");

  NameSupply namesupply{"prefix"};
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName_1
  LOG_PRINT_VAR(namesupply->ReserveName("prefix_TestName_1"));
  LOG_PRINT_VAR(namesupply->ContainsName("TestName"));
}

}  // namespace name_supply_test

REGISTER_TEST_SUITE(name_supply_test::IrNameSupplyTest,
                    ir_name_supply_test_IrNameSupplyTest);
