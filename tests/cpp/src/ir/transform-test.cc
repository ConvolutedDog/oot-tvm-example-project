#include "ir/transform-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace transform_test {

using tvm::IntImm;
using tvm::PrimExpr;
using tvm::runtime::Array;
using tvm::tir::Buffer;
using tvm::tir::For;
using tvm::tir::Stmt;

std::ostream &operator<<(std::ostream &os,
                         const Map<String, Map<String, String>> &configs) {
  os << "\n";
  for (const auto &outerpair : configs) {
    String outerkey = outerpair.first;
    os << "  " << outerkey << ": {";
    for (const auto &innerpair : outerpair.second) {
      String innerkey = innerpair.first;
      String innervalue = innerpair.second;
      os << innerkey << ": " << innervalue;
    }
    os << "}\n";
  }
  return os;
}

/// @brief `PassContext::ListConfigs()` returns all of the configs registered for pass.
/// The configs are registered using the `TVM_REGISTER_PASS_CONFIG_OPTION` macro. For
/// example, the following code registers the config `relax.FuseOps.max_depth` as an
/// integer, which means the maximum number of operations in one fused function when
/// TVM performs the pass `relax.FuseOps`.
///     TVM_REGISTER_PASS_CONFIG_OPTION("relax.FuseOps.max_depth", Integer);
/// When using this config like:
///     GetConfig("relax.FuseOps.max_depth", Integer(kMaxFusedOps));
/// it defaults to `kMaxFusedOps` if this config is not set.
/// Users can use `PassContext::RegisterConfigOption` to register a config option.
void IrPassContextTest() {
  LOG_SPLIT_LINE("IrPassContextTest");

  // Print the current configs.
  Map<String, Map<String, String>> configs = PassContext::ListConfigs();
  LOG_PRINT_VAR("PassContext::ListConfigs():");
  LOG_PRINT_VAR(configs);

  // Register a new config option.
  PassContext::RegisterConfigOption<tvm::Integer>("xxxxxxxxxxxxxx-testIntegerConfig");
  LOG_PRINT_VAR("PassContext::ListConfigs():");
  LOG_PRINT_VAR(PassContext::ListConfigs());

  PassContext passctx = PassContext::Create();
  LOG_PRINT_VAR(passctx);
  LOG_PRINT_VAR(passctx->diag_ctx);
  LOG_PRINT_VAR(passctx->config);
  LOG_PRINT_VAR(passctx->make_traceable);
  LOG_PRINT_VAR(passctx->num_evals);
  LOG_PRINT_VAR(passctx->tuning_api_database);

  LOG_PRINT_VAR(PassContext::Current());

  // Get the registered test config.
  auto testIntegerConfig = passctx->GetConfig<tvm::Integer>(
      "xxxxxxxxxxxxxx-testIntegerConfig", tvm::Integer{128});
  LOG_PRINT_VAR(testIntegerConfig);
}

void IrPassTest() {
  LOG_SPLIT_LINE("IrPassTest");

  // Create an IRModule.
  GlobalVar globalvar("globalvar");

  // NOLINTNEXTLINE
  TensorStructInfo scalar_struct_info{ShapeExpr{{PrimExpr{-1}}}, DataType::Int(32, 1)};
  Var gv0{"gv0", scalar_struct_info};
  Var f{
      "f", FuncStructInfo{{scalar_struct_info}, scalar_struct_info}
  };
  Var ipt{"ipt", scalar_struct_info};
  Var x0{"x0", scalar_struct_info};
  Var x1{"x1", scalar_struct_info};
  Var x2{"x2", scalar_struct_info};
  Var y{"y", scalar_struct_info};

  NDArray ndarrayconstvalue2 =
      NDArray::Empty(ShapeTuple{1}, DLDataType{DLDataTypeCode::kDLInt, 32, 1},
                     DLDevice{DLDeviceType::kDLCPU, 0});
  int constvalue2 = 2;
  ndarrayconstvalue2.CopyFromBytes(&constvalue2, sizeof(int));

  NDArray ndarrayconstvalue1 =
      NDArray::Empty(ShapeTuple{1}, DLDataType{DLDataTypeCode::kDLInt, 32, 1},
                     DLDevice{DLDeviceType::kDLCPU, 0});
  int constvalue1 = 1;
  ndarrayconstvalue1.CopyFromBytes(&constvalue1, sizeof(int));

  Constant constant2{ndarrayconstvalue2};
  Constant constant1{ndarrayconstvalue1};
  LOG_PRINT_VAR(GetShapeOf(constant2));
  LOG_PRINT_VAR(GetShapeOf(constant1));
  // NOLINTNEXTLINE
  BindingBlock inner_block{
      {VarBinding{x0, constant2}, VarBinding{y, Call{f, {x0}}}}
  };
  LOG_PRINT_VAR(inner_block);
  // NOLINTNEXTLINE
  Function inner_func{{ipt}, SeqExpr({inner_block}, x0), scalar_struct_info};
  LOG_PRINT_VAR(inner_func);

  OpRegistry *opregistry = OpRegistry::Global();
  // NOLINTNEXTLINE
  Call call_add{
      Op::Get("relax.add"), {x1, Call{f, {x1}}}
  };
  LOG_PRINT_VAR(call_add);
  // NOLINTNEXTLINE
  BindingBlock outer_block{
      {VarBinding{f, inner_func}, VarBinding{x1, constant1}, VarBinding{x2, call_add},
       VarBinding{gv0, x2}}
  };

  Function func{{}, SeqExpr({outer_block}, gv0), scalar_struct_info};

  // Build IRModule
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);
  LOG_SPLIT_LINE("");

  // Apply the pass.
  LOG_PRINT_VAR(CallTIRRewrite()->Info());
  LOG_PRINT_VAR(LowerTVMBuiltin()->Info());
  LOG_PRINT_VAR(CallTIRRewrite()(irmodule2));
  LOG_PRINT_VAR(LowerTVMBuiltin()(irmodule2));

  Sequential passes{
      {CallTIRRewrite(), LowerTVMBuiltin()}
  };
  LOG_PRINT_VAR(passes.operator->()->operator()(irmodule2, PassContext::Current()));

  LOG_PRINT_VAR(PrintIR()(irmodule2));
}

/// @todo @Benkang Peng
/// 1. Correctly instantiate `relax.call_tir` and `relax.assert_op`
/// 2. test the function of Pass `CallTIRRewrite` and `LowerTVMBuiltin`
void IrPassTest2() {

  TensorStructInfo tensorInfo{ShapeExpr({tvm::Integer(8)}), DataType::Float(32)};

  Var x{"x", tensorInfo};
  Var gv0("gv0", tensorInfo);

  PrimExpr n = 8;
  Buffer bufA = tvm::tir::decl_buffer({n}, DataType::Float(32), "a");
  Buffer bufB = tvm::tir::decl_buffer({n}, DataType::Float(32), "b");
  Buffer bufC = tvm::tir::decl_buffer({n}, DataType::Float(32), "c");

  LOG_PRINT_VAR(bufA);

  tvm::tir::Var i("i", DataType::Int(32));
  Stmt body = For(
      /*loop_var*/ i,
      /*min*/ 0,
      /*extent*/ n,
      /*kind*/ tvm::tir::ForKind::kSerial,
      /*body*/
      tvm::tir::BufferStore(
          bufC, tvm::tir::BufferLoad(bufA, {i}) + tvm::tir::BufferLoad(bufB, {i}), {i}));

  LOG_PRINT_VAR(body);

  tvm::tir::PrimFunc tirFunc(
      /*args*/ Array<tvm::tir::Var>({bufA->data, bufB->data, bufC->data}),
      /*body*/ body,
      /*ret_type*/ tvm::VoidType());
  LOG_PRINT_VAR(tirFunc);

  GlobalVar myTirFunc("my_tir_func");
  Array<Expr> args = {myTirFunc, x, gv0};

  /// don't know how to pass the `args` into `relax.call_tir`
  /// `args` passed into `relax.call_tir` are wrong, which cause `callTir` can't be print
  Call callTir(Op::Get("relax.call_tir"), args);
  LOG_PRINT_VAR(callTir->op);
  LOG_PRINT_VAR(callTir->args);
  // LOG_PRINT_VAR(callTir);//have bug

  Call assertCall(Op::Get("relax.assert_op"), {x});
  LOG_PRINT_VAR(assertCall->op);
  LOG_PRINT_VAR(assertCall->args);
  // LOG_PRINT_VAR(assertCall);//have bug

  BindingBlock block({VarBinding(gv0, callTir), VarBinding(x, assertCall)});
  // LOG_PRINT_VAR(block);//have bug

  Function func(
      /*params*/ {x},
      /*body*/ SeqExpr({block}, gv0),
      /*ret_struct_info*/ tensorInfo);

  IRModule mod;
  mod->Add(myTirFunc, tirFunc);
  mod->Add(GlobalVar("main"), func);
  // LOG_PRINT_VAR(mod);
}
}  // namespace transform_test

REGISTER_TEST_SUITE(transform_test::IrPassContextTest,
                    ir_transform_test_IrPassContextTest);
REGISTER_TEST_SUITE(transform_test::IrPassTest, ir_transform_test_IrPassTest);

REGISTER_TEST_SUITE(transform_test::IrPassTest2, ir_transform_test_IrPassTest2)