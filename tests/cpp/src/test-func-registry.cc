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

void TestSuiteRegistry::PrintAllTestSuiteNames(std::ostream &os) const {
  os << "All test suites: {";
  for (auto &kv : Global()->suites_) {
    os << kv.first << ", ";
  }
  os << "}\n";
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
