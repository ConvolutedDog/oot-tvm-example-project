#include "node/object-path-test.h"
#include "test-func-registry.h"

namespace object_path_test {

void ObjectPathTest() {
  LOG_SPLIT_LINE("ObjectPathTest");

  /// @todo (yangjianchao) I don't understand what this is for at the moment, I'll add
  /// more later.
  LOG_PRINT_VAR(ObjectPath::Root()->Length());
}

}  // namespace object_path_test

void ObjectPathTest() { object_path_test::ObjectPathTest(); }

namespace {

REGISTER_TEST_SUITE(ObjectPathTest);

}  // namespace
