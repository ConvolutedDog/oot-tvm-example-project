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
  registry->RunTestSuite("runtime_ndarray_test_RuntimeNDArrayTest");
  // Bug on MacOS
  registry->RunTestSuite("runtime_inplace_array_base_test_RuntimeInplaceArrayBaseTest");
  registry->RunTestSuite("runtime_object_test_RuntimeObjectTest");
  registry->RunTestSuite("runtime_objectref_test_RuntimeObjectRefTest");
  registry->RunTestSuite("ir_pass_test_IrPassTestTemp");
  registry->RunTestSuite("ir_expr_test_IrPrimExprTest");
  registry->RunTestSuite("ir_expr_test_IrBoolTest");
  registry->RunTestSuite("ir_expr_test_IrIntegerTest");
  registry->RunTestSuite("ir_expr_test_IrRangeTest");
  registry->RunTestSuite("ir_type_test_IrPrimTypeTest");
  registry->RunTestSuite("ir_type_test_IrPointerTypeTest");
  registry->RunTestSuite("ir_type_test_IrTupleTypeTest");
  registry->RunTestSuite("ir_type_test_IrFuncTypeTest");
  registry->RunTestSuite("node_reflection_test_NodeAttrVisitorTest");
  registry->RunTestSuite("node_reflection_test_NodeReflectionVTableTest");
  registry->RunTestSuite("ir_attrs_test_IrAttrUtilsTests");
  registry->RunTestSuite("ir_attrs_test_IrAttrFieldInfoTest");
  registry->RunTestSuite("ir_attrs_test_IrAttrsTest");
  registry->RunTestSuite("ir_attrs_test_IrDictAttrsTest");
  registry->RunTestSuite("ir_function_test_IrBaseFuncTest");
  registry->RunTestSuite("ir_source_map_test_IrSpanTest");
  registry->RunTestSuite("ir_source_map_test_IrSourceTest");
  registry->RunTestSuite("node_functor_test_NodeNodeFunctorTest");
  registry->RunTestSuite("ir_type_functor_test_IrTypeFunctorTest");
  registry->RunTestSuite("runtime_tvmpodvalue_test_RuntimeTvmPodValueTest");
  registry->RunTestSuite("ir_op_test_IrOpNodeTest");
  registry->RunTestSuite("ir_op_test_IrOpTest");
  registry->RunTestSuite("relax_expr_test_RelaxCallTest");
  registry->RunTestSuite("relax_expr_test_RelaxTupleTest");
  registry->RunTestSuite("relax_expr_test_RelaxTupleGetItemTest");
  registry->RunTestSuite("relax_expr_test_RelaxLeafExprTest");
  registry->RunTestSuite("relax_expr_test_RelaxBindTest");
  registry->RunTestSuite("ir_module_test_IrModuleTest");
  registry->RunTestSuite("target_target_kind_test_TargetTargetKindTest");
  registry->RunTestSuite("target_target_test_TargetTargetTest");
  registry->RunTestSuite("ir_global_info_test_IrGlobalInfoTest");
  registry->RunTestSuite("ir_global_info_test_IrVDeviceTest");
  registry->RunTestSuite("ir_global_info_test_IrDummyGlobalInfoTest");
  registry->RunTestSuite("ir_analysis_test_IrAnalysisTest");
  registry->RunTestSuite("ir_diagnostic_test_IrDiagnosticTest");
  registry->RunTestSuite("ir_diagnostic_test_IrDiagnosticContextTest");
  registry->RunTestSuite("ir_name_supply_test_IrNameSupplyTest");
  registry->RunTestSuite("ir_global_var_supply_test_IrGlobalVarSupplyTest");
  registry->RunTestSuite("ir_replace_global_vars_test_IrReplaceGlobalVarsTest");
  registry->RunTestSuite("ir_transform_test_IrPassContextTest");
  registry->RunTestSuite("ir_transform_test_IrPassTest");
  registry->RunTestSuite("node_repr_printer_test_NodeDumpTest");
  registry->RunTestSuite("node_repr_printer_test_NodeAsLegacyReprTest");
  registry->RunTestSuite("node_repr_printer_test_NodeReprPrinterTest");
  registry->RunTestSuite("node_repr_printer_test_NodeReprLegacyPrinterTest");
  registry->RunTestSuite("node_object_path_test_NodeObjectPathTest");
  registry->RunTestSuite("tir_var_test_TirVarTest");
  registry->RunTestSuite("tir_var_test_TirSizeVarTest");
  registry->RunTestSuite("tir_var_test_TirIterVarTest");
  registry->RunTestSuite("tir_buffer_test_TirBufferTest");
  registry->RunTestSuite("tir_buffer_test_TirDataProducerTest");
  registry->RunTestSuite("tir_expr_test_TirExprTest");
  registry->RunTestSuite("tir_expr_test_TirBufferLoadTest");
  registry->RunTestSuite("tir_expr_test_TirProducerLoadTest");
  registry->RunTestSuite("tir_expr_test_TirRampTest");
  registry->RunTestSuite("tir_expr_test_TirBroadcastTest");
  registry->RunTestSuite("tir_expr_test_TirLetTest");
  registry->RunTestSuite("tir_expr_test_TirCallTest");
  registry->RunTestSuite("tir_expr_test_TirShuffleTest");
  registry->RunTestSuite("tir_expr_test_TirCommReducerTest");
  registry->RunTestSuite("tir_expr_test_TirReduceTest");
  registry->RunTestSuite("tir_op_test_TirOpTest");
  registry->RunTestSuite("tir_expr_functor_test_ExprFunctorTest");
  registry->RunTestSuite("tir_stmt_test_TirBufferStoreTest");
  registry->RunTestSuite("tir_stmt_test_TirProducerStoreTest");
  registry->RunTestSuite("tir_stmt_test_TirAllocateTest");
  registry->RunTestSuite("tir_stmt_test_TirForTest");
  registry->RunTestSuite("tir_stmt_test_TirPrefetchTest");
  registry->RunTestSuite("tir_stmt_test_TirTypeAnnotationTest");
  registry->RunTestSuite("tir_function_test_TirPrimFuncTest");
  registry->RunTestSuite("tir_function_test_TirTensorIntrinTest");
  registry->RunTestSuite("tir_function_test_TirSpecialize");
  registry->RunTestSuite("node_serialization_test_NodeSerializationTest");
}

int main() {
  TestMethod2();
  return 0;
}
