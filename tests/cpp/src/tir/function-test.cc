#include "tir/function-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/container/variant.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

namespace function_test {

using tvm::PointerType;
using tvm::PrimType;
using tvm::Range;
using tvm::runtime::make_object;
using tvm::runtime::Map;
using tvm::runtime::Variant;
using tvm::te::BufferLoad;
using tvm::te::BufferStore;
using tvm::tir::Block;
using tvm::tir::BlockRealize;
using tvm::tir::BufferRegion;
using tvm::tir::BufferType;
using tvm::tir::Call;
using tvm::tir::Evaluate;
using tvm::tir::For;
using tvm::tir::ForKind;
using tvm::tir::IfThenElse;
using tvm::tir::IterVar;
using tvm::tir::IterVarType;
using tvm::tir::Specialize;
using tvm::tir::TensorIntrinNode;
void TirPrimFuncTest() {
  LOG_SPLIT_LINE("TirPrimFuncTest");

  /// PrimFuncNode contains TIR statements.
  Var m("m", DataType::Int(32));
  Var n("n", DataType::Int(32));
  int lanes = 2;
  DataType dtype = DataType::BFloat(16, lanes);
  TensorType retTy{2, dtype};
  Array<PrimExpr> shape{m, n};
  Tensor tensor{
      shape, dtype, PlaceholderOp{"placeholder", shape, dtype},
        0
  };
  PrimExpr value{0};
  Broadcast broadcast{value, lanes};
  Array<PrimExpr> indices = {m, n};
  ProducerStore producerstore{tensor, broadcast, indices};
  ProducerLoad producerload(tensor, indices);
  /// @bug The generated python code cannot execute.
  PrimFunc primfunc{
      {m, n},
      producerstore, retTy
  };
  LOG_PRINT_VAR(primfunc);
  /// # from tvm.script import tir as T
  /// # from tvm.script import relax as R
  ///
  /// @T.prim_func(private=True)
  /// def main(m: T.int32, n: T.int32) -> R.Tensor(ndim=2, dtype="bfloat16x2"):
  ///     placeholder[m, n] = T.Broadcast(0, 2)

  LOG_PRINT_VAR(primfunc->func_type_annotation());
  /// Output:
  ///   I.FuncType([T.int32, T.int32], R.Tensor(ndim=2, dtype="bfloat16x2"))

  LOG_PRINT_VAR(primfunc->params);         // [m, n]
  LOG_PRINT_VAR(primfunc->body);           // m = T.int32()
                                           // n = T.int32()
                                           // placeholder[m, n] = T.Broadcast(0, 2)
  LOG_PRINT_VAR(primfunc->ret_type);       // R.Tensor(ndim=2, dtype="bfloat16x2")
  LOG_PRINT_VAR(primfunc->buffer_map);     // {}
  LOG_PRINT_VAR(primfunc->checked_type_);  // Same to func_type_annotation()
  LOG_PRINT_VAR(primfunc->struct_info_);   // R.Callable((R.Prim("int32"),
                                           // R.Prim("int32")), R.Object, True)
}

Stmt BuildMatmulBody(Buffer &A, Buffer &B, Buffer &C) {
  Var i("i"), j("j"), k("k");

  PrimExpr c_ij = BufferLoad(C, {i, j});
  PrimExpr new_value = c_ij + BufferLoad(A, {i, k}) * BufferLoad(B, {k, j});

  Stmt store = BufferStore(C, new_value, {i, j});

  Stmt k_loop = For(
      /*loop_var=*/k,
      /*min=*/0,
      /*extent=*/16,
      /*for_type=*/ForKind::kSerial,
      /*body=*/store);

  Stmt j_loop = For(
      /*loop_var=*/j,
      /*min=*/0,
      /*extent=*/16,
      /*for_type=*/ForKind::kSerial,
      /*body=*/k_loop);

  Stmt i_loop = For(
      /*loop_var=*/i,
      /*min=*/0,
      /*extent=*/16,
      /*for_type=*/ForKind::kSerial,
      /*body=*/j_loop);

  return i_loop;
}

/// @brief Define the computation logic of GEMM function.
PrimFunc DefineMatmulDesc() {
  Var A_handle("A_handle", PointerType(PrimType(DataType::Int(32))));
  Var B_handle("B_handle", PointerType(PrimType(DataType::Int(32))));
  Var C_handle("C_handle", PointerType(PrimType(DataType::Int(32))));

  DataType dtype(DataType::Int(32));
  Buffer A(A_handle, dtype, {16, 16}, {16, 1}, 0, "A", 4, 0, BufferType::kDefault, {});
  Buffer B(B_handle, dtype, {16, 16}, {16, 1}, 0, "B", 4, 0, BufferType::kDefault, {});
  Buffer C(C_handle, dtype, {16, 16}, {16, 1}, 0, "C", 4, 0, BufferType::kDefault, {});

  Var i("i"), j("j"), k("k");

  Stmt blockRealize = BlockRealize(
      {i, j, k}, tvm::Bool(true),
      Block(
          /*Array<IterVar>*/ {IterVar(Range(0, 16), i, IterVarType::kDataPar),
                              IterVar(Range(0, 16), j, IterVarType::kDataPar),
                              IterVar(Range(0, 16), k, IterVarType::kCommReduce)},
          /*read*/
          {BufferRegion::FullRegion(A), BufferRegion::FullRegion(B)},
          /*write*/ {BufferRegion::FullRegion(C)},
          /*name_hint*/ "matmul",
          /*body*/ BuildMatmulBody(A, B, C)));

  LOG_PRINT_VAR(blockRealize);
  /// map the buffer to the handle.
  Map<Var, Buffer> bufferMap;
  bufferMap.Set(A_handle, A);
  bufferMap.Set(B_handle, B);
  bufferMap.Set(C_handle, C);

  return PrimFunc({A_handle, B_handle, C_handle}, blockRealize, {}, bufferMap);
}

PrimFunc DefineMatmulImpl() {
  Var A_handle("A_handle", PointerType(PrimType(DataType::Int(32))));
  Var B_handle("B_handle", PointerType(PrimType(DataType::Int(32))));
  Var C_handle("C_handle", PointerType(PrimType(DataType::Int(32))));

  DataType dtype(DataType::Int(32));
  Buffer A(A_handle, dtype, {16, 16}, {16, 1}, 0, "A", 4, 0, BufferType::kDefault, {});
  Buffer B(B_handle, dtype, {16, 16}, {16, 1}, 0, "B", 4, 0, BufferType::kDefault, {});
  Buffer C(C_handle, dtype, {16, 16}, {16, 1}, 0, "C", 4, 0, BufferType::kDefault, {});

  CHECK(A->data.defined()) << "A->data is null!";
  CHECK(B->data.defined()) << "B->data is null!";
  CHECK(C->data.defined()) << "C->data is null!";

  /// the computation logic related to hardware
  tvm::relax::ExternFunc extFunc("wmma.mma.sync.aligned");

  Stmt body = Evaluate(Call(DataType::Void(), extFunc,
                            {C->data, C->elem_offset, A->data, A->elem_offset, B->data,
                             B->elem_offset, C->data, C->elem_offset}));

  /// map the buffer to the handle.
  Map<Var, Buffer> bufferMap;
  bufferMap.Set(A_handle, A);
  bufferMap.Set(B_handle, B);
  bufferMap.Set(C_handle, C);

  return PrimFunc({A_handle, B_handle, C_handle}, body, {}, bufferMap);
}
/// @todo (yangjianchao)

void TirTensorIntrinTest() {
  LOG_SPLIT_LINE("TirTensorIntrinTest");

  PrimFunc matmulDesc = DefineMatmulDesc();
  LOG_PRINT_VAR(matmulDesc);

  PrimFunc matmulImpl = DefineMatmulImpl();

  /// @bug can't print `matmulImpl`
  // LOG_PRINT_VAR(matmulImpl);  //

  LOG_PRINT_VAR(matmulImpl->params);
  // LOG_PRINT_VAR(matmulImpl->body);///@bug

  TensorIntrin matmulIntrin(matmulDesc, matmulImpl);
  TensorIntrin::Register("cuda mma", matmulIntrin);

  LOG_PRINT_VAR(TensorIntrin::Get("cuda mma"));
}

/// @todo (yangjianchao)
void TirSpecialize() {
  LOG_SPLIT_LINE("TirSpecialize");

  Var m("m"), n("n");
  Var A_handle("A_handle", PointerType(PrimType(DataType::Int(32))));
  Var B_handle("B_handle", PointerType(PrimType(DataType::Int(32))));

  DataType dtype(DataType::Int(32));
  Buffer A(A_handle, dtype, {m, n}, {16, 1}, 0, "A", 4, 0, BufferType::kDefault, {});
  Buffer B(B_handle, dtype, {m, n}, {16, 1}, 0, "B", 4, 0, BufferType::kDefault, {});

  Var i("i"), j("j");
  Stmt j_loop =
      For(j, 0, n, ForKind::kSerial, BufferStore(B, BufferLoad(A, {i, j}), {i, j}));

  Stmt i_loop = For(i, 0, m, ForKind::kSerial, j_loop);

  LOG_PRINT_VAR(i_loop);

  /// map the buffer to the handle.
  Map<Var, Buffer> bufferMap;
  bufferMap.Set(A_handle, A);
  bufferMap.Set(B_handle, B);

  PrimFunc func({A_handle, B_handle, m, n}, i_loop, tvm::VoidType(), bufferMap);

  LOG_PRINT_VAR(func);
  LOG_PRINT_VAR(func->params);

  m = func->params[2];
  n = func->params[3];

  /// specialize the size of the buffer
  Map<Var, Variant<Buffer, PrimExpr>> newMap;
  newMap.Set(m, PrimExpr(128));
  newMap.Set(n, PrimExpr(128));

  PrimFunc newFunc = Specialize(func, newMap);

  LOG_PRINT_VAR(newFunc);
}

}  // namespace function_test

REGISTER_TEST_SUITE(function_test::TirPrimFuncTest, tir_function_test_TirPrimFuncTest);
REGISTER_TEST_SUITE(function_test::TirTensorIntrinTest,
                    tir_function_test_TirTensorIntrinTest);
REGISTER_TEST_SUITE(function_test::TirSpecialize, tir_function_test_TirSpecialize);
