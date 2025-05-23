#include "tir/block-scope-test.h"
#include "test-func-registry.h"

namespace block_scope_test {

void TirStmtSRefTest() { LOG_SPLIT_LINE("TirStmtSRefTest"); }

void TirSRefTreeCreatorTest() { LOG_SPLIT_LINE("TirSRefTreeCreatorTest"); }

void TirDependencyTest() { LOG_SPLIT_LINE("TirDependencyTest"); }

void TirBlockScopeTest() { LOG_SPLIT_LINE("TirBlockScopeTest"); }

}  // namespace block_scope_test

REGISTER_TEST_SUITE(block_scope_test::TirStmtSRefTest,
                    tir_block_scope_test_TirStmtSRefTest);
REGISTER_TEST_SUITE(block_scope_test::TirSRefTreeCreatorTest,
                    tir_block_scope_test_TirSRefTreeCreatorTest);
REGISTER_TEST_SUITE(block_scope_test::TirDependencyTest,
                    tir_block_scope_test_TirDependencyTest);
REGISTER_TEST_SUITE(block_scope_test::TirBlockScopeTest,
                    tir_block_scope_test_TirBlockScopeTest);
