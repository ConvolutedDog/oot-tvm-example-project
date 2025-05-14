#include "ir/name-supply-test.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace name_supply_test {

void NameSupplyTest() {
  LOG_SPLIT_LINE("NameSupplyTest");

  NameSupply namesupply{"prefix"};
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName
  LOG_PRINT_VAR(namesupply->FreshName("TestName"));  // prefix_TestName_1
  LOG_PRINT_VAR(namesupply->ReserveName("prefix_TestName_1"));
  LOG_PRINT_VAR(namesupply->ContainsName("TestName"));
}

}  // namespace name_supply_test

void NameSupplyTest() { name_supply_test::NameSupplyTest(); }
