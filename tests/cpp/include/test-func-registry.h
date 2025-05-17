#ifndef TEST_FUNC_REGISTRY
#define TEST_FUNC_REGISTRY

#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"

/// Macro for printing variable name and its value.
#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';

#define LOG_BLANK_LINE std::cout << '\n'

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

/// Macros for concatenating strings.
#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

/// Macro for getting the third argument.
#define GET_MACRO_TEST_SUITE(_1, _2, Name, ...) Name

/// @brief Macro for registering test suites with the global TestSuiteRegistry. It is
/// overloaded to support 1 or 2 arguments.
///
/// @note An anonymous namespace `namespace{}` ensures that variables/functions are only
/// visible within the current file, preventing symbol conflicts across different files.
///
/// @note For example:
///   🚀 1 argument: `REGISTER_TEST_SUITE(TestSuite1)` will be expand️ed to:
///     @code {.cpp}
///     namespace {
///     GET_MACRO_TEST_SUITE(TestSuite1, REGISTER_TEST_SUITE_2,
///                          REGISTER_TEST_SUITE_1)(TestSuite1);
///     }
///     @endcode
///     Then this will be further replaced by `REGISTER_TEST_SUITE_1(TestSuite1)`.
///   🚀 2 argument:
///     `REGISTER_TEST_SUITE(TestSuite2, NameHint)` will be expand️ed to:
///     @code {.cpp}
///     namespace {
///     REGISTER_TEST_SUITE_2(TestSuite2, NameHint);
///     }
///     @endcode
///
/// Usage:
/// 1. Define a function in a .cc/.cpp file:
///      namespace a {
///        void TestSuiteName() {
///          ...
///        }
///      }  // namespace a
/// 2. Register it with the global TestSuiteRegistry in the same .cc/.cpp file:
///      REGISTER_TEST_SUITE(::a::TestSuiteName);
///    And, to avoid the name conflict, we can use a second param `key` (std::string)
///    to register the function:
///      REGISTER_TEST_SUITE(::a::TestSuiteName, JustPlaceSomeKeyHere);
///
///    For example, `REGISTER_TEST_SUITE(ndarray_test::RuntimeNDArrayTest,
///                                      runtime_ndarray_test_RuntimeNDArrayTest);`
///    will be expanded to:
///
///    @code {.cpp}
///    namespace {
///    ::test_func_registry::TestSuiteRegistry __make_TestSuite0 =            \
///     ::test_func_registry::TestSuiteRegistry::Global()                      \
///         -> RegisterTestSuite(runtime_ndarray_test_RuntimeNDArrayTest,      \
///              -> RegisterTestSuite(#key, func, __FILE__, __LINE__)          \
///    }
///    @endcode
/// 3. Call the function from main.cc:
///    int main() {
///      // Run a single test suite.
///        // If registered without a specified key.
///        TestSuiteRegistry::Global()->RunTestSuite("TestSuiteName");
///        // If registered with a specified key.
///        TestSuiteRegistry::Global()->RunTestSuite("JustPlaceSomeKeyHere");
///      // Run all test suites.
///      TestSuiteRegistry::Global()->RunAllTestSuites();
///    }
#define REGISTER_TEST_SUITE(...)                                                         \
  namespace {                                                                            \
  GET_MACRO_TEST_SUITE(__VA_ARGS__, REGISTER_TEST_SUITE_2,                               \
                       REGISTER_TEST_SUITE_1)(__VA_ARGS__);                              \
  }

/// @brief Version with 1 argument (func only, key defaults to func).
#define REGISTER_TEST_SUITE_1(func) REGISTER_TEST_SUITE_2(func, func)

/// @brief Version with 2 arguments (func and custom key).
/// @note The variable inside the anonymous namespace is initialized before the `main()`
/// function executes. This property is leveraged to perform registration during static
/// initialization: the variable's constructor includes the registration logic which add
/// entries into global registry table (`TestSuiteRegistry::Global()->suites_`), enabling
/// automatic registration without explicit includings and calls.
#define REGISTER_TEST_SUITE_2(func, key)                                                 \
  ::test_func_registry::TestSuiteRegistry &STR_CONCAT(__make_TestSuite, __COUNTER__) =   \
      ::test_func_registry::TestSuiteRegistry::Global()                                  \
          -> RegisterTestSuite(#key, func, __FILE__, __LINE__)
