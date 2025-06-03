#include "ir/diagnostic-test.h"
#include "test-func-registry.h"

namespace diagnostic_test {

inline const char *DiagnosticLevel2String(DiagnosticLevel lv) {
  switch (lv) {
    case DiagnosticLevel::kBug: return "Bug";
    case DiagnosticLevel::kError: return "Error";
    case DiagnosticLevel::kWarning: return "Warning";
    case DiagnosticLevel::kNote: return "Note";
    case DiagnosticLevel::kHelp: return "Help";
  }
  return "Unknown";
}

void IrDiagnosticTest() {
  LOG_SPLIT_LINE("IrDiagnosticTest");
  Diagnostic diagnostic{
      DiagnosticLevel::kBug, Span{SourceName::Get("diagnostic_test.cc"), 1, 2, 3, 4},
      "IrDiagnosticTest - kBug"
  };
  LOG_PRINT_VAR(diagnostic);
  LOG_PRINT_VAR(DiagnosticLevel2String(diagnostic->level));
  LOG_PRINT_VAR(diagnostic->span);
  LOG_PRINT_VAR(diagnostic->loc);
  LOG_PRINT_VAR(diagnostic->message);

  DiagnosticBuilder builder{diagnostic->level, diagnostic->span};
  LOG_PRINT_VAR(builder.operator Diagnostic()->message);
  LOG_PRINT_VAR(((Diagnostic)builder)->message);
  LOG_PRINT_VAR(static_cast<Diagnostic>(builder)->span);

  LOG_PRINT_VAR(Diagnostic::Bug(diagnostic->span).operator Diagnostic());
}

void IrDiagnosticContextTest() {
  LOG_SPLIT_LINE("IrDiagnosticContextTest");

  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  /// Create tvm::relax::Function
  Expr opexpr = tvm::Op::Get("relax.add");
  Var arg1{
      "arg1", tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4}
  };
  Var arg2{
      "arg2", tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4}
  };
  Call call{
      opexpr, {arg1, arg2}
  };
  Function func{
      {arg1,                     arg2},
      call,
      tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4   },
      true,
  };

  /// @note TVMScript cannot print functions of type: BaseFunc
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);
  LOG_SPLIT_LINE("");

  DiagnosticContext diagctx{irmodule2, DiagnosticRenderer{[](DiagnosticContext ctx) {
                              Diagnostic diag{DiagnosticLevel::kBug, Span{}, ""};
                              ctx.Emit(diag);
                              LOG_SPLIT_LINE("Render Start");
                              LOG_PRINT_VAR(diag->span);
                              LOG_PRINT_VAR(diag->loc);
                              LOG_PRINT_VAR(DiagnosticLevel2String(diag->level));
                              LOG_PRINT_VAR(diag->message);
                              LOG_SPLIT_LINE("Render End");
                            }}};
  diagctx.Render();
}

}  // namespace diagnostic_test

REGISTER_TEST_SUITE(diagnostic_test::IrDiagnosticTest,
                    ir_diagnostic_test_IrDiagnosticTest);
REGISTER_TEST_SUITE(diagnostic_test::IrDiagnosticContextTest,
                    ir_diagnostic_test_IrDiagnosticContextTest);
