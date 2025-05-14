#include "test-func-registry.h"
#include "tvm/runtime/logging.h"

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

void TestSuiteRegistry::PrintAllTestSuiteNames() const {
  std::cout << "All test suites: {";
  for (auto &kv : Global()->suites_) {
    std::cout << kv.first << ", ";
  }
  std::cout << "}\n";
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

void TestSuiteRegistry::RunTestSuite(const String &name) const {
  ICHECK(Global()->suites_.count(name) != 0)
      << "TestSuite " << name << " is not registered";
  std::cout << "⭕⭕⭕ Running TestSuite <" << name << "> located at "
            << Global()->span_[name] << "...\n";
  Global()->suites_[name]();
  std::cout << "✅✅✅ TestSuite <" << name << "> passed\n\n";
}

void TestSuiteRegistry::RunAllTestSuites() const {
  for (auto &kv : Global()->suites_) {
    RunTestSuite(kv.first);
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
