#ifndef TEST_FUNC_REGISTRY
#define TEST_FUNC_REGISTRY

#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace test_func_registry {

using tvm::runtime::Array;
using tvm::runtime::String;

using TestFuncTy = void (*)();

struct Source {
  String name;
  int line;
};

std::ostream &operator<<(std::ostream &os, const Source &src);

class TestSuiteRegistry {
private:
  static TestSuiteRegistry *inst;

public:
  static TestSuiteRegistry *Global();

  Array<String> ListAllTestSuiteNames() const;

  void PrintAllTestSuiteNames() const;

  TestSuiteRegistry &RegisterTestSuite(const String &name, TestFuncTy func, String source,
                                       int line);

  void RunTestSuite(const String &name) const;

  void RunAllTestSuites() const;

private:
  struct InstanceDeleter;

  std::unordered_map<String, TestFuncTy> suites_;  // NOLINT
  std::unordered_map<String, Source> span_;        // NOLINT
  static InstanceDeleter deleter_;                 // NOLINT
};

}  // namespace test_func_registry

#endif  // TEST_FUNC_REGISTRY

using TestSuiteRegistry = test_func_registry::TestSuiteRegistry;

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

#define REGISTER_TEST_SUITE(func)                                                        \
  static void __test_suite_##func() { func(); }                                          \
  static ::test_func_registry::TestSuiteRegistry &STR_CONCAT(__make_TestSuite,           \
                                                             __COUNTER__) =              \
      ::test_func_registry::TestSuiteRegistry::Global()                                  \
          -> RegisterTestSuite(#func, __test_suite_##func, __FILE__, __LINE__)
