#include "test-func-registry.h"
#include "tvm/runtime/logging.h"
#include <algorithm>
#include <iomanip>
#include <sys/ioctl.h>  // for ioctl() and TIOCGWINSZ
#include <unistd.h>     // for isatty()
#include <vector>

namespace test_func_registry {

std::ostream &operator<<(std::ostream &os, const Source &src) {
  return os << src.name << ":L" << src.line;
}

TestSuiteRegistry *TestSuiteRegistry::Global() {
  if (inst == nullptr)
    inst = new TestSuiteRegistry();
  return inst;
}

Array<String> TestSuiteRegistry::ListAllTestSuiteNames() const {
  Array<String> ret;
  for (auto &kv : Global()->suites_) {
    ret.push_back(kv.first);
  }
  return ret;
}

void TestSuiteRegistry::PrintAllTestSuiteNames(std::ostream &os) const {
  // Get the width of terminal.
  struct winsize w;
  int terminalWidth = 80;
  if (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
    terminalWidth = w.ws_col;
  }

  os << "All test suites:\n";

  std::vector<std::string> names;
  for (auto &kv : Global()->suites_) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());

  size_t maxNameWidth = 0;
  for (const auto &name : names) {
    maxNameWidth = std::max(maxNameWidth, name.size());
  }
  maxNameWidth += 2;

  size_t columns = std::max(1, static_cast<int>(terminalWidth / maxNameWidth));
  size_t rows = (names.size() + columns - 1) / columns;

  os << BLUE_TEXT;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < columns; ++col) {
      size_t index = col * rows + row;  // Col-major
      if (index < names.size()) {
        os << std::left << std::setw(maxNameWidth) << names[index];
      }
    }
    os << "\n";
  }
  os << RESET_TEXT;
}

TestSuiteRegistry &TestSuiteRegistry::RegisterTestSuite(const String &name,
                                                        TestFuncTy func, String source,
                                                        int line) {
  ICHECK(Global()->suites_.count(name) == 0)
      << "TestSuite " << name << " already registered";
  ICHECK(func != nullptr) << "TestSuite " << name << " is nullptr";
  Global()->suites_[name] = func;
  Global()->span_[name] = {std::move(source), line};
  return *Global();
}

void TestSuiteRegistry::RunTestSuite(const String &name, std::ostream &os) const {
  ICHECK(Global()->suites_.count(name) != 0)
      << "TestSuite " << name << " is not registered";
  os << "⭕⭕⭕ Running TestSuite <" << name << "> located at " << Global()->span_[name]
     << "...\n";
  Global()->suites_[name]();
  os << "✅✅✅ TestSuite <" << name << "> passed\n\n";
}

void TestSuiteRegistry::RunAllTestSuites(std::ostream &os) const {
  for (auto &kv : Global()->suites_) {
    RunTestSuite(kv.first, os);
  }
}

struct TestSuiteRegistry::InstanceDeleter {
  ~InstanceDeleter() {
    if (inst != nullptr) {
      delete inst;
      inst = nullptr;
    }
  }
};

TestSuiteRegistry *TestSuiteRegistry::inst = nullptr;
TestSuiteRegistry::InstanceDeleter TestSuiteRegistry::deleter_;

}  // namespace test_func_registry
