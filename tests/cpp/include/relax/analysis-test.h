#include "utils.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

namespace analysis_test {
using tvm::DataType;
using tvm::PrimExpr;
using tvm::With;
using tvm::arith::Analyzer;
using tvm::relax::BaseCheckResult;
using tvm::relax::CanProveShapeEqual;
using tvm::relax::GetStaticType;
using tvm::relax::ObjectStructInfo;
using tvm::relax::PrimStructInfo;
using tvm::relax::ShapeExpr;
using tvm::relax::ShapeStructInfo;
using tvm::relax::StructInfo;
using tvm::relax::StructInfoBaseCheck;
using tvm::relax::StructInfoBaseCheckPrecondition;
using tvm::relax::StructInfoFromType;
using tvm::relax::TensorStructInfo;
using tvm::relax::TensorType;
using tvm::runtime::Map;
using tvm::runtime::Registry;
void AnalysisTest();
void BaseCheckTest();

}  // namespace analysis_test