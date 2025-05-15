#ifndef TEST_FUNC_REGISTRY
#define TEST_FUNC_REGISTRY

#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"

/// Macro for printing variable name and its value.
#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';

/// Macro for printing a separator line with custom stmt.
#define SPLIT_L(num_equal) std::string(num_equal, '=') << " "
#define SPLIT_R(num_equal) " " << std::string(num_equal, '=') << '\n'
#define LOG_SPLIT_LINE_IMPL(stmt, num_equal)                                             \
  std::cout << SPLIT_L(num_equal) << (stmt) << SPLIT_R(num_equal)
#define LOG_SPLIT_LINE(stmt) LOG_SPLIT_LINE_IMPL(stmt, 14)

namespace test_func_registry {

using tvm::runtime::Array;
using tvm::runtime::String;

/// Type alias for test function pointer (no arguments, `void` return). Currently, the
/// registration of test functions only allows this type of function pointer, but if
/// needed in the future, we will add more.
using TestFuncTy = void (*)();

/// @brief Struct representing source code location (file name and line number). This will
/// be used when registering a test function and can be used to print the source code
/// location when a test starts.
struct Source {
  String name;
  int line;
};

/// @brief Overloaded output operator for `Source` struct.
std::ostream &operator<<(std::ostream &os, const Source &src);

/// @brief This class implements a lightweight reflection mechanism and has good
/// scalability. It achieves reflection functionality for test functions through the
/// registry pattern and macros, while maintaining the efficiency of C++. This class
/// maintains a singleton instance for managing test suite registration and execution.
class TestSuiteRegistry {
private:
  // Singleton instance.
  static TestSuiteRegistry *inst;

public:
  // Gets the global singleton instance (creates if doesn't exist).
  static TestSuiteRegistry *Global();

  // Returns an array of all registered test suite names.
  Array<String> ListAllTestSuiteNames() const;

  // Prints all registered test suite names to `std::ostream &os`.
  void PrintAllTestSuiteNames(std::ostream &os = std::cout) const;

  // Registers a new test suite with given name, function pointer, and source location.
  TestSuiteRegistry &RegisterTestSuite(const String &name, TestFuncTy func, String source,
                                       int line);

  // Runs a specific test suite by name.
  void RunTestSuite(const String &name, std::ostream &os = std::cout) const;

  // Runs all registered test suites.
  void RunAllTestSuites(std::ostream &os = std::cout) const;

private:
  // Helper class for automatic cleanup of singleton instance. We maintain a static
  // instance of `InstanceDeleter` in `TestSuiteRegistry`, thus the destructor of
  // `InstanceDeleter` will be called defaultly when the program exits. The destructor
  // of `InstanceDeleter` will delete the singleton instance of `TestSuiteRegistry`
  // to avoid memory leak.
  struct InstanceDeleter;

  // Map of test names to functions.
  std::unordered_map<String, TestFuncTy> suites_;  // NOLINT
  // Map of test names to source locations.
  std::unordered_map<String, Source> span_;  // NOLINT
  // Static deleter instance.
  static InstanceDeleter deleter_;  // NOLINT
};

}  // namespace test_func_registry

#endif  // TEST_FUNC_REGISTRY

/// Type alias for easier access to TestSuiteRegistry.
using TestSuiteRegistry = test_func_registry::TestSuiteRegistry;

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

/// Macro for registering test suites:
/// 1. Creates a static wrapper function.
/// 2. Registers it with the global TestSuiteRegistry.
///
/// Usage:
/// 1. Define a function without prefix namespace in a .cc/.cpp file:
///    void TestSuiteName() {
///      ::namespace_x::namespace_y:: ... ::TestSuiteFunc();
///    }
/// 2. Register it with the global TestSuiteRegistry in the same .cc/.cpp file:
///    namespace {
///      REGISTER_TEST_SUITE(TestSuiteName);
///    }
/// 3. Call the function from main.cc:
///    int main() {
///      // Run a single test suite.
///      TestSuiteRegistry::Global()->RunTestSuite("TestSuiteName");
///      // Run all test suites.
///      TestSuiteRegistry::Global()->RunAllTestSuites();
///    }
#define REGISTER_TEST_SUITE(func)                                                        \
  static void __test_suite_##func() { func(); }                                          \
  static ::test_func_registry::TestSuiteRegistry &STR_CONCAT(__make_TestSuite,           \
                                                             __COUNTER__) =              \
      ::test_func_registry::TestSuiteRegistry::Global()                                  \
          -> RegisterTestSuite(#func, __test_suite_##func, __FILE__, __LINE__)