#include "tir/op-test.h"
#include "test-func-registry.h"
#include <cmath>
#include <tvm/ir/expr.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace op_test {

void OpTest() {
  LOG_SPLIT_LINE("OpTest");

  /// GetType
  LOG_PRINT_VAR(GetType(1));
  LOG_PRINT_VAR(GetType(1.0f));
  LOG_PRINT_VAR(GetType(tvm::floor(2.0f)));

  /// GetTypeFromRuntimeDataType
  LOG_PRINT_VAR(GetTypeFromRuntimeDataType(tvm::DataType::BFloat(16, 4)));
  LOG_PRINT_VAR(GetTypeFromRuntimeDataType(tvm::DataType::Float(32)));
  LOG_PRINT_VAR(GetTypeFromRuntimeDataType(tvm::DataType::Int(32, 16)));

  /// GetRuntimeDataType
  LOG_PRINT_VAR(
      GetRuntimeDataType(GetTypeFromRuntimeDataType(tvm::DataType::BFloat(16, 4))));

  /// ret
  ///
  /// @brief `ret(PrimExpr value)` is used to get the return value of `value`:
  /// namespace tir::builtin {
  ///   const Op& ret() {
  ///     static const Op& op = Op ::Get(
  ///         "tir."
  ///         "ret");
  ///     return op;
  ///   }
  /// }
  /// PrimExpr ret(PrimExpr value, Span span) {
  ///   CHECK(value.defined());
  ///   return tir::Call(value.dtype(), tir::builtin::ret(), {value}, span);
  /// }
  LOG_PRINT_VAR(ret(1));
  LOG_PRINT_VAR(tvm::floor(2.0f));

  /// max_value & min_value & infinity
  LOG_PRINT_VAR(max_value(tvm::DataType::BFloat(16, 1)));  // Lanes must be 1.
  LOG_PRINT_VAR(min_value(tvm::DataType::BFloat(16, 1)));
  /// @note `infinity(...)` can only used by DataType::kFloat type with lanes = 1 and bits
  /// = 16/32/64.
  LOG_PRINT_VAR(infinity(
      tvm::DataType::Float(16, 1)));  // Cannot decide infinity for type bfloat16.

  /// cast & reinterpret
  ///
  /// @brief
  /// If the target type of `cast` is scalar (lanes = 1):
  ///   1. `cast` can only be used by `value`s of `IntImm` or `FloatImm` type, and will
  ///       use `make_const` to return for these two types.
  ///   2. otherwise it returns a `tvm::tir::Cast` node.
  /// Elseif the target type of `cast` is scalable (lanes < -1) or fixed length vector
  /// (lanes > 1), it is to say `value` is a vector:
  ///   1. the target type `t` should also be scalable if `value` is a scalable vector.
  ///   2. the target type `t` should have same `vscale_factor()` if `value` is a scalable
  ///      vector.
  ///   3. the target type `t` should have same lanes if `value` is a fixed length vector.
  ///   4. if `value` is a `tvm::tir::Broadcast` node:
  ///        return Broadcast(cast(vtype, broadcast->value, span), broadcast->lanes,
  ///                         span);
  ///      elif `value` is a `tvm::tir::Ramp` node, `t` is DataType::kInt/DataType::UkInt:
  ///        return tir::Ramp(cast(vtype, ramp->base, span), cast(vtype, ramp->stride,
  ///        span),
  ///                         ramp->lanes, span);
  ///      else:
  ///        return tir::Cast(t, value, span);
  /// Else:  // the target type of `cast` is not scalable (lanes < -1) or fixed length
  ///        // vector (lanes > 1).
  ///   @todo (yangjianchao) Not commonly used.
  LOG_PRINT_VAR(cast(tvm::DataType::Float(32, 1), 1));
  /// `reinterpret(...)` returns `tir::Call(t, tir::builtin::reinterpret(), {value})`.
  LOG_PRINT_VAR(reinterpret(tvm::DataType::Float(32, 1), 1));
  LOG_PRINT_VAR(
      reinterpret(tvm::DataType::Float(16, 1), max_value(tvm::DataType::BFloat(16, 1))));

  /// add ~ abs
  LOG_PRINT_VAR(tvm::add(tvm::PrimExpr{1}, cast(tvm::DataType::Float(32, 1), 1)));
  LOG_PRINT_VAR(tvm::abs(infinity(tvm::DataType::Float(16, 1))));
  LOG_PRINT_VAR(tvm::abs(cast(tvm::DataType::Float(32, 1), -1)));

  /// isnan ~ isinf
  LOG_PRINT_VAR(isnan(tvm::PrimExpr{NAN}));
  LOG_PRINT_VAR(isfinite(INFINITY));
  LOG_PRINT_VAR(isfinite(1));
  LOG_PRINT_VAR(isinf(infinity(tvm::DataType::Float(16, 1))));

  /// sum ~ prod
  int n = 10;
  IterVar i = IterVar(Range(0, n), Var{"i"}, IterVarType::kCommReduce);
  IterVar j = IterVar(Range(0, n), Var{"j"}, IterVarType::kCommReduce);
  Array<IterVar> rdom = {i, j};
  Array<tvm::PrimExpr> init = {make_const(tvm::DataType::Int(32), 0)};
  Buffer buffer = decl_buffer({n, n}, tvm::DataType::Int(32), "A");
  tvm::PrimExpr source = BufferLoad(buffer, {i, j});
  // $totalsum = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} A[i][j]$.
  tvm::PrimExpr totalsum = sum(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalsum);
  tvm::PrimExpr totalmin = min(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalmin);
  tvm::PrimExpr totalmax = max(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalmax);
  tvm::PrimExpr totalprod = prod(source, rdom, init, tvm::Span());
  LOG_PRINT_VAR(totalprod);

  Array<tvm::PrimExpr> initbool = {make_const(tvm::DataType::Bool(1), false)};
  Buffer bufferbool = decl_buffer({n, n}, tvm::DataType::Bool(1), "Abool");
  tvm::PrimExpr sourcebool = BufferLoad(bufferbool, {i, j});

  tvm::PrimExpr totalall = all(sourcebool, rdom, initbool, tvm::Span());
  LOG_PRINT_VAR(totalall);

  tvm::PrimExpr totalany = any(sourcebool, rdom, initbool, tvm::Span());
  LOG_PRINT_VAR(totalany);

  /// floor ~ trunc
  LOG_PRINT_VAR(floor(1.2f));
  LOG_PRINT_VAR(trunc(1.2f));
  LOG_PRINT_VAR(round(1.2f));
  LOG_PRINT_VAR(ceil(1.2f));
  LOG_PRINT_VAR(nearbyint(1.2f));
  LOG_PRINT_VAR(nearbyint(1.5f));
  LOG_PRINT_VAR(nearbyint(1.6f));

  /// LargeUIntImm: Construct a large uint constant by its low 32 bits and
  /// high 32bits.
  LOG_PRINT_VAR(LargeUIntImm(tvm::DataType::Int(32), 128, 128));

  /// IsPointerType: Check if type is a pointer to a runtime element type.
  LOG_PRINT_VAR(IsPointerType(tvm::PointerType{tvm::PrimType{tvm::DataType::Float(32)}},
                              tvm::DataType::Float(32)));

  /// make_const & make_zero & const_true & const_false
  LOG_PRINT_VAR(make_const(tvm::DataType::Float(32), 1.0f));
  LOG_PRINT_VAR(make_zero(tvm::DataType::Float(32)));
  LOG_PRINT_VAR(const_true(4));
  LOG_PRINT_VAR(const_false(4));

  /// as_const_int
  LOG_PRINT_VAR(*as_const_int(make_const(tvm::DataType::UInt(32), 1)));

  /// is_const_int & is_no_op & is_one & is_zero & is_const_number
  LOG_PRINT_VAR(is_const_int(make_const(tvm::DataType::Int(32), 1)));
  LOG_PRINT_VAR(is_const_int(make_const(tvm::DataType::Int(32), 1), 1));
  LOG_PRINT_VAR(is_const_int(make_const(tvm::DataType::Int(32), 1), 2));
  LOG_PRINT_VAR(is_const_int(1));
  LOG_PRINT_VAR(is_no_op(tvm::tir::Stmt{}));
  LOG_PRINT_VAR(is_one(make_const(tvm::DataType::Int(32), 1)));
  LOG_PRINT_VAR(is_zero(make_const(tvm::DataType::Int(32), 0)));
  LOG_PRINT_VAR(is_const_number(2));
  LOG_PRINT_VAR(is_const_number(tvm::tir::Mul{2, 3}));
  LOG_PRINT_VAR(is_const_number(make_const(tvm::DataType::Int(32), 1)));

  /// foldl: Left fold.
  /// @todo (yangjianchao) foldl(...)

  /// MakeConstScalar
  // clang-format off
  /// @note If `t` is DataType::Int, returns `IntImm`. If `t` is DataType::UInt, only when
  /// 0 <= `value` < static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), it returns
  /// `IntImm`, and if `value` >= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
  /// it returns `tvm::tir::Call`. If `t` is DataType::BFloat/Float, returns `FloatImm`.
  // clang-format on
  LOG_PRINT_VAR(MakeConstScalar(tvm::DataType::Float(32), 1.0f));
  LOG_PRINT_VAR(MakeConstScalar(tvm::DataType::Float(32), true));
  LOG_PRINT_VAR(MakeConstScalar(tvm::DataType::Float(32), false));

  /// make_const
  ///
  /// @note For scalar type, use `MakeConstScalr()` to return IntImm/UIntImm/FloatImm/...,
  /// for vector type, returns `tir::Broadcast` node.
  LOG_PRINT_VAR(make_const(tvm::DataType::Float(32), 1.0f));

  /// make_zero
  ///
  /// @note If `t` is a DataType::Handle, returns `tvm::tir::Call` node, otherwise it will
  /// call `make_const`.
  LOG_PRINT_VAR(make_zero(tvm::DataType::Float(32)));

  /// @todo (yangjianchao) The rest of them can be used as you go, and they are
  /// all helper functions, including q_multiply_shift, fast_erf_float_expr, exp ~ clz,
  /// atan2 ~ ldexp, is_const_power_of_two_integer ~ is_no_op, and all of the operators.
}

}  // namespace op_test

REGISTER_TEST_SUITE(op_test::OpTest, tir_op_test_TirOpTest);
