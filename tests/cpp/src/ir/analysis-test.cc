#include "ir/analysis-test.h"
#include "test-func-registry.h"
#include "utils.h"
#include <tvm/ir/op.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

namespace analysis_test {

using tvm::Op;
using tvm::Optional;
using tvm::relax::ShapeExprNode;
using tvm::relax::StructInfo;
using tvm::relax::TensorStructInfo;
using tvm::relax::Tuple;
using tvm::relax::TupleStructInfo;
using tvm::runtime::Array;

Expr MakeCallTIR(Expr func, Tuple args, Array<TensorStructInfo> out_sinfo_list,
                 Optional<Expr> packed_ints) {
  for (const TensorStructInfo &sinfo : out_sinfo_list) {
    const auto *shape = sinfo->shape.as<ShapeExprNode>();
    CHECK(shape != nullptr)
        << "out_sinfo of call_tir should have defined ShapeExpr as shape. "
           "However, one given structure info is "
        << sinfo;
  }

  StructInfo out_sinfo{nullptr};
  if (out_sinfo_list.size() == 1) {
    out_sinfo = out_sinfo_list[0];
  } else {
    out_sinfo = TupleStructInfo({out_sinfo_list.begin(), out_sinfo_list.end()});
  }

  static const Op &op = Op::Get("relax.call_tir");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_sinfo});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_sinfo});
  }
  return call;
}

void IrAnalysisTest() {
  LOG_SPLIT_LINE("IrAnalysisTest");

  /// @brief List all op names.
  Array<tvm::runtime::String> names = ListAllOpNames();
  AdjustScreenPrint(std::cout, names);

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

  LOG_PRINT_VAR(CollectCallMap(irmodule2));

  /// @todo
  // using tirVar = tvm::tir::Var;
  // using tvm::tir::PrimFunc;
  // auto calltir = MakeCallTIR(
  //     PrimFunc{
  //         {
  //          tirVar{"x"},
  //          },
  //         tvm::tir::Evaluate{0}
  // },
  //     Tuple{{Expr{GlobalVar{"y"}}}}, {}, tvm::NullOpt);
  // LOG_PRINT_VAR(calltir);
}

}  // namespace analysis_test

REGISTER_TEST_SUITE(analysis_test::IrAnalysisTest, ir_analysis_test_IrAnalysisTest);
