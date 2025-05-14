#include "ir/analysis-test.h"
#include "ir/attrs-test.h"
#include "ir/diagnostic-test.h"
#include "ir/expr-test.h"
#include "ir/function-test.h"
#include "ir/global-info-test.h"
#include "ir/global-var-supply-test.h"
#include "ir/module-test.h"
#include "ir/name-supply-test.h"
#include "ir/op-test.h"
#include "ir/pass-test.h"
#include "ir/replace-global-vars-test.h"
#include "ir/source-map-test.h"
#include "ir/transform-test.h"
#include "ir/type-functor-test.h"
#include "ir/type-test.h"
#include "node/functor-test.h"
#include "node/reflection-test.h"
#include "relax/expr-test.h"
#include "runtime/inplacearraybase-test.h"
#include "runtime/ndarrayutils-test.h"
#include "runtime/object-test.h"
#include "runtime/tvmpodvalue-test.h"
#include "target/target-kind-test.h"
#include "target/target-test.h"
#include "test-func-registry.h"
#include "tir/var-test.h"
#include <tvm/runtime/logging.h>

void TestMethod1() {
  /// Test suite registry
  TestSuiteRegistry *registry = TestSuiteRegistry::Global();

  /// Print all test suite names.
  registry->PrintAllTestSuiteNames();

  /// Method 1: Run all test suites.
  registry->RunAllTestSuites();
}

[[deprecated("Deprecated and will not be maintained, please use TestMethod3() instead.")]]
void TestMethod2() {
  /// Test suite registry
  TestSuiteRegistry *registry = TestSuiteRegistry::Global();

  /// Print all test suite names.
  registry->PrintAllTestSuiteNames();

  /// Method 2: Run each specified test suite manually.
  std::cout << "Running NDArrayTest...\n";
  tvm::runtime::NDArrayTest();
  std::cout << "NDArrayTest passed!\n\n";

  std::cout << "Running AutoSchedulerTest...\n";
  InplaceArrayBaseTest();  // Bug on MacOS
  std::cout << "AutoSchedulerTest passed!\n\n";

  std::cout << "Running ObjectTest...\n";
  ObjectTest();
  std::cout << "ObjectTest passed!\n\n";

  std::cout << "Running ObjectRefTest...\n";
  ObjectRefTest();
  std::cout << "ObjectRefTest passed!\n\n";

  std::cout << "Running PassTest...\n";
  PassTestTemp();
  std::cout << "PassTest passed!\n\n";

  std::cout << "Running ExprTest...\n";
  PrimExprTest();
  BoolTest();
  IntegerTest();
  RangeTest();
  std::cout << "ExprTest passed!\n\n";

  std::cout << "Running TypeTest...\n";
  PrimTypeTest();
  PointerTypeTest();
  TupleTypeTest();
  FuncTypeTest();
  std::cout << "TypeTest passed!\n\n";

  std::cout << "Running VarTest...\n";
  VarTest();
  SizeVarTest();
  IterVarTest();
  std::cout << "VarTest passed!\n\n";

  std::cout << "Running ReflectionTest...\n";
  AttrVisitorTest();
  ReflectionVTableTest();
  std::cout << "ReflectionTest passed!\n\n";

  std::cout << "Running AttrTests...\n";
  AttrUtilsTests();
  AttrFieldInfoTest();
  AttrsTest();
  DictAttrsTest();
  std::cout << "AttrTests passed!\n\n";

  std::cout << "Running BaseFuncTest...\n";
  BaseFuncTest();
  std::cout << "BaseFuncTest passed!\n\n";

  std::cout << "Running SourceMapTest...\n";
  SpanTest();
  SourceTest();
  std::cout << "SourceMapTest passed!\n\n";

  std::cout << "Running NodeFunctorTest...\n";
  NodeFunctorTest();
  std::cout << "NodeFunctorTest passed!\n\n";

  std::cout << "Running TypeFunctorTest...\n";
  TypeFunctorTest();
  std::cout << "TypeFunctorTest passed!\n\n";

  std::cout << "Running TvmPodValueTest...\n";
  TvmPodValueTest();
  std::cout << "TvmPodValueTest passed!\n\n";

  std::cout << "Running OpTest...\n";
  OpNodeTest();
  OpTest();
  std::cout << "OpTest passed!\n\n";

  std::cout << "Running RelaxExprTest...\n";
  CallTest();
  TupleTest();
  TupleGetItemTest();
  LeafExprTest();
  BindTest();
  std::cout << "RelaxExprTest passed!\n\n";

  std::cout << "Running ModuleTest...\n";
  ModuleTest();
  std::cout << "ModuleTest passed!\n\n";

  std::cout << "Running TargetTest...\n";
  TargetKindTest();
  std::cout << "TargetTest passed!\n\n";

  std::cout << "Running TargetTest...\n";
  TargetTest();
  std::cout << "TargetTest passed!\n\n";

  std::cout << "Running GlobalInfoTest...\n";
  GlobalInfoTest();
  VDeviceTest();
  DummyGlobalInfoTest();
  std::cout << "GlobalInfoTest passed!\n\n";

  std::cout << "Running AnalysisTest...\n";
  AnalysisTest();
  std::cout << "AnalysisTest passed!\n\n";

  std::cout << "Running DiagnosticTest...\n";
  DiagnosticTest();
  DiagnosticContextTest();
  std::cout << "DiagnosticTest passed!\n\n";

  std::cout << "Running NameSupplyTest...\n";
  NameSupplyTest();
  std::cout << "NameSupplyTest passed!\n\n";

  std::cout << "Running GlobalVarSupplyTest...\n";
  GlobalVarSupplyTest();
  std::cout << "GlobalVarSupplyTest passed!\n\n";

  std::cout << "Running ReplaceGlobalVarsTest...\n";
  ReplaceGlobalVarsTest();
  std::cout << "ReplaceGlobalVarsTest passed!\n\n";

  std::cout << "Running PassTest...\n";
  PassContextTest();
  PassTest();
  std::cout << "PassTest passed!\n\n";

  std::cout << "All tests passed!\n";
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
  registry->RunTestSuite("PassContextTest");
  registry->RunTestSuite("PassTest");
}

int main() {
  TestMethod3();
  return 0;
}
