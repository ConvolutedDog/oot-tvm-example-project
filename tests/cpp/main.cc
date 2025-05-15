#include "deprecated.h"
#include "test-func-registry.h"
#include <tvm/runtime/logging.h>

void TestMethod1() {
  /// Test suite registry
  TestSuiteRegistry *registry = TestSuiteRegistry::Global();

  /// Print all test suite names.
  registry->PrintAllTestSuiteNames();

  /// Method 1: Run all test suites.
  registry->RunAllTestSuites();
}

void TestMethod2() {
  /// Method 2: Run each specified test suite manually.
  deprecated::TestMethod2();
}

void TestMethod3() {
  /// Test suite registry
  TestSuiteRegistry *registry = TestSuiteRegistry::Global();

  /// Print all test suite names.
  registry->PrintAllTestSuiteNames();

  /// Method 3: Run each specific test suite.
  // registry->RunTestSuite("NDArrayTest");
  // registry->RunTestSuite("InplaceArrayBaseTest");  // Bug on MacOS
  // registry->RunTestSuite("ObjectTest");
  // registry->RunTestSuite("ObjectRefTest");
  // registry->RunTestSuite("PassTestTemp");
  // registry->RunTestSuite("PrimExprTest");
  // registry->RunTestSuite("BoolTest");
  // registry->RunTestSuite("IntegerTest");
  // registry->RunTestSuite("RangeTest");
  // registry->RunTestSuite("PrimTypeTest");
  // registry->RunTestSuite("PointerTypeTest");
  // registry->RunTestSuite("TupleTypeTest");
  // registry->RunTestSuite("FuncTypeTest");
  // registry->RunTestSuite("VarTest");
  // registry->RunTestSuite("SizeVarTest");
  // registry->RunTestSuite("IterVarTest");
  // registry->RunTestSuite("AttrVisitorTest");
  // registry->RunTestSuite("ReflectionVTableTest");
  // registry->RunTestSuite("AttrUtilsTests");
  // registry->RunTestSuite("AttrFieldInfoTest");
  // registry->RunTestSuite("AttrsTest");
  // registry->RunTestSuite("DictAttrsTest");
  // registry->RunTestSuite("BaseFuncTest");
  // registry->RunTestSuite("SpanTest");
  // registry->RunTestSuite("SourceTest");
  // registry->RunTestSuite("NodeFunctorTest");
  // registry->RunTestSuite("TypeFunctorTest");
  // registry->RunTestSuite("TvmPodValueTest");
  // registry->RunTestSuite("OpNodeTest");
  // registry->RunTestSuite("OpTest");
  // registry->RunTestSuite("CallTest");
  // registry->RunTestSuite("TupleTest");
  // registry->RunTestSuite("TupleGetItemTest");
  // registry->RunTestSuite("LeafExprTest");
  // registry->RunTestSuite("BindTest");
  // registry->RunTestSuite("ModuleTest");
  // registry->RunTestSuite("TargetKindTest");
  // registry->RunTestSuite("TargetTest");
  // registry->RunTestSuite("GlobalInfoTest");
  // registry->RunTestSuite("VDeviceTest");
  // registry->RunTestSuite("DummyGlobalInfoTest");
  // registry->RunTestSuite("AnalysisTest");
  // registry->RunTestSuite("DiagnosticTest");
  // registry->RunTestSuite("DiagnosticContextTest");
  // registry->RunTestSuite("NameSupplyTest");
  // registry->RunTestSuite("GlobalVarSupplyTest");
  // registry->RunTestSuite("ReplaceGlobalVarsTest");
  // registry->RunTestSuite("PassContextTest");
  // registry->RunTestSuite("PassTest");
  registry->RunTestSuite("DumpTest");
  registry->RunTestSuite("AsLegacyReprTest");
  registry->RunTestSuite("ReprPrinterTest");
  registry->RunTestSuite("ReprLegacyPrinterTest");
}

int main() {
  TestMethod3();
  return 0;
}
