#include "ir/analysis-test.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace analysis_test {

void AnalysisTest() {
  LOG_SPLIT_LINE("AnalysisTest");

  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  /// Create tvm::relax::Function
  Expr opexpr = tvm::Op::Get("relax.nn.conv2d");
  Var arg1{"arg1", tvm::relax::ShapeStructInfo{4}};
  Var arg2{"arg2", tvm::relax::ShapeStructInfo{4}};
  Call call{
      opexpr, {arg1, arg2}
  };
  Function func{
      {arg1, arg2},
      call,
      tvm::relax::ShapeStructInfo{4},
      true,
  };

  /// @note TVMScript cannot print functions of type: BaseFunc
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);
  LOG_SPLIT_LINE("");

  LOG_PRINT_VAR(CollectCallMap(irmodule2));
}

}  // namespace analysis_test

void AnalysisTest() { analysis_test::AnalysisTest(); }
