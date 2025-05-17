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
  /// Test suite registry
  TestSuiteRegistry *registry = TestSuiteRegistry::Global();

  /// Print all test suite names.
  registry->PrintAllTestSuiteNames();

  /// Method 2: Run each specific test suite.
  registry->RunTestSuite("runtime_ndarray_test_NDArrayTest");
  // Bug on MacOS
  registry->RunTestSuite("runtime_inplace_array_base_test_InplaceArrayBaseTest");
  registry->RunTestSuite("runtime_object_test_ObjectTest");
  registry->RunTestSuite("runtime_objectref_test_ObjectRefTest");
  registry->RunTestSuite("ir_pass_test_PassTestTemp");
  registry->RunTestSuite("ir_expr_test_PrimExprTest");
  registry->RunTestSuite("ir_expr_test_BoolTest");
  registry->RunTestSuite("ir_expr_test_IntegerTest");
  registry->RunTestSuite("ir_expr_test_RangeTest");
  registry->RunTestSuite("ir_type_test_PrimTypeTest");
  registry->RunTestSuite("ir_type_test_PointerTypeTest");
  registry->RunTestSuite("ir_type_test_TupleTypeTest");
  registry->RunTestSuite("ir_type_test_FuncTypeTest");
  registry->RunTestSuite("node_reflection_test_AttrVisitorTest");
  registry->RunTestSuite("node_reflection_test_ReflectionVTableTest");
  registry->RunTestSuite("ir_attrs_test_AttrUtilsTests");
  registry->RunTestSuite("ir_attrs_test_AttrFieldInfoTest");
  registry->RunTestSuite("ir_attrs_test_AttrsTest");
  registry->RunTestSuite("ir_attrs_test_DictAttrsTest");
  registry->RunTestSuite("ir_function_test_BaseFuncTest");
  registry->RunTestSuite("ir_source_map_test_SpanTest");
  registry->RunTestSuite("ir_source_map_test_SourceTest");
  registry->RunTestSuite("node_functor_test_NodeFunctorTest");
  registry->RunTestSuite("ir_type_functor_test_TypeFunctorTest");
  registry->RunTestSuite("runtime_tvmpodvalue_test_TvmPodValueTest");
  registry->RunTestSuite("ir_op_test_OpNodeTest");
  registry->RunTestSuite("ir_op_test_OpTest");
  registry->RunTestSuite("relax_expr_test_CallTest");
  registry->RunTestSuite("relax_expr_test_TupleTest");
  registry->RunTestSuite("relax_expr_test_TupleGetItemTest");
  registry->RunTestSuite("relax_expr_test_LeafExprTest");
  registry->RunTestSuite("relax_expr_test_BindTest");
  registry->RunTestSuite("ir_module_test_ModuleTest");
  registry->RunTestSuite("target_target_kind_test_TargetKindTest");
  registry->RunTestSuite("target_target_test_TargetTest");
  registry->RunTestSuite("ir_global_info_test_GlobalInfoTest");
  registry->RunTestSuite("ir_global_info_test_VDeviceTest");
  registry->RunTestSuite("ir_global_info_test_DummyGlobalInfoTest");
  registry->RunTestSuite("ir_analysis_test_AnalysisTest");
  registry->RunTestSuite("ir_diagnostic_test_DiagnosticTest");
  registry->RunTestSuite("ir_diagnostic_test_DiagnosticContextTest");
  registry->RunTestSuite("ir_name_supply_test_NameSupplyTest");
  registry->RunTestSuite("ir_global_var_supply_test_GlobalVarSupplyTest");
  registry->RunTestSuite("ir_replace_global_vars_test_ReplaceGlobalVarsTest");
  registry->RunTestSuite("ir_transform_test_PassContextTest");
  registry->RunTestSuite("ir_transform_test_PassTest");
  registry->RunTestSuite("node_repr_printer_test_DumpTest");
  registry->RunTestSuite("node_repr_printer_test_AsLegacyReprTest");
  registry->RunTestSuite("node_repr_printer_test_ReprPrinterTest");
  registry->RunTestSuite("node_repr_printer_test_ReprLegacyPrinterTest");
  registry->RunTestSuite("node_object_path_test_ObjectPathTest");
  registry->RunTestSuite("tir_var_test_VarTest");
  registry->RunTestSuite("tir_var_test_SizeVarTest");
  registry->RunTestSuite("tir_var_test_IterVarTest");
  registry->RunTestSuite("tir_buffer_test_BufferTest");
  registry->RunTestSuite("tir_buffer_test_DataProducerTest");
  registry->RunTestSuite("tir_expr_test_TirExprTest");
  registry->RunTestSuite("tir_expr_test_BufferLoadTest");
  registry->RunTestSuite("tir_expr_test_ProducerLoadTest");
  registry->RunTestSuite("tir_expr_test_RampTest");
  registry->RunTestSuite("tir_expr_test_BroadcastTest");
  registry->RunTestSuite("tir_expr_test_LetTest");
  registry->RunTestSuite("tir_expr_test_TirCallTest");
  registry->RunTestSuite("tir_expr_test_ShuffleTest");
  registry->RunTestSuite("tir_expr_test_CommReducerTest");
  registry->RunTestSuite("tir_expr_test_ReduceTest");
}

int main() {
  TestMethod2();
  return 0;
}
