#include "relax/analysis-test.h"
#include "utils.h"
#include <cassert>
#include <tvm/relax/analysis.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/data_type.h>

namespace analysis_test {

void AnalysisTest() {
  PrimExpr l1(1), l2(2.0f);
  PrimExpr r1(1), r2(2.0f);

  Analyzer ana = Analyzer();
  LOG_PRINT_VAR(CanProveShapeEqual({l1, l2}, {r1, r2}, &ana));

  tvm::tir::Var m("m", DataType::Int(64));
  tvm::tir::Var n("n", DataType::Int(64));
  ShapeExpr shape({m, n});
  auto sinfo = TensorStructInfo(shape, DataType::Int(32));

  auto type1 = tvm::relax::GetStaticType(sinfo);
  TensorType type2(2, DataType::Int(32));

  auto StructuralEqual = Registry::Get("node.StructuralEqual");
  assert((*StructuralEqual)(type1, type2, true, false));

  LOG_PRINT_VAR(StructInfoFromType(type1));

  Map<tvm::tir::Var, PrimExpr> shape_map{
      {m, 128},
      {n, 128}
  };
  auto erased = EraseToWellDefined(sinfo, shape_map, {});
  LOG_PRINT_VAR(erased);
}

void BaseCheckTest() {
  ObjectStructInfo obj0;

  PrimStructInfo prim0(DataType::Int(32));
  PrimStructInfo prim1(DataType::Float(32));

  ShapeStructInfo shape0(-1);
  ShapeStructInfo shape1({2, 2});

  tvm::tir::Var m("m", DataType::Int(64)), n("n", DataType::Int(64));
  ShapeStructInfo shape2({m, n});

  TensorStructInfo tensor0(DataType::Int(32), -1);
  TensorStructInfo tensor1(DataType::Int(32), 2);

  assert(StructInfoBaseCheck(prim0, prim1) == BaseCheckResult::kFailL0);
  assert(StructInfoBaseCheck(prim0, obj0) == BaseCheckResult::kFailL1);
  assert(StructInfoBaseCheck(shape2, shape1) == BaseCheckResult::kFailL2);
  assert(StructInfoBaseCheck(prim0, prim0) == BaseCheckResult::kPass);

  LOG_PRINT_VAR(StructInfoBaseCheckPrecondition(shape2, shape1));

  /// @todo @BenkangPeng The rest of `analysis.h` are not implemented yet.
}
}  // namespace analysis_test

REGISTER_TEST_SUITE(analysis_test::AnalysisTest, relax_analysis_test_AnalysisTest);
REGISTER_TEST_SUITE(analysis_test::BaseCheckTest, relax_analysis_test_BaseCheckTest);
