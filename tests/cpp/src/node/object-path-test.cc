#include "node/object-path-test.h"
#include "test-func-registry.h"

namespace object_path_test {

void NodeObjectPathTest() {
  LOG_SPLIT_LINE("NodeObjectPathTest");

  /// @todo (yangjianchao) I don't understand what this is for at the moment, I'll add
  /// more later.
  LOG_PRINT_VAR(ObjectPath::Root()->Length());
}

}  // namespace object_path_test

REGISTER_TEST_SUITE(object_path_test::NodeObjectPathTest,
                    node_object_path_test_NodeObjectPathTest);
