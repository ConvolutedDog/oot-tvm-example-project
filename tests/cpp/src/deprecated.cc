#include "deprecated.h"

namespace deprecated {

void TestMethod2() {
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

}  // namespace deprecated
