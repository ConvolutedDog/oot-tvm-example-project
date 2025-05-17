#include "tvm/tir/op.h"
#include <functional>
#include <tvm/tir/var.h>

namespace op_test {

using tvm::abs;
using tvm::add;
using tvm::bitwise_and;
using tvm::bitwise_neg;
using tvm::bitwise_or;
using tvm::bitwise_xor;
using tvm::cast;
using tvm::ceildiv;
using tvm::div;
using tvm::equal;
using tvm::floordiv;
using tvm::floormod;
using tvm::GetRuntimeDataType;
using tvm::GetType;
using tvm::GetTypeFromRuntimeDataType;
using tvm::greater;
using tvm::greater_equal;
using tvm::if_then_else;
using tvm::indexdiv;
using tvm::indexmod;
using tvm::infinity;
using tvm::less;
using tvm::less_equal;
using tvm::likely;
using tvm::logaddexp;
using tvm::logical_and;
using tvm::logical_not;
using tvm::logical_or;
using tvm::max;
using tvm::max_value;
using tvm::min;
using tvm::min_value;
using tvm::mul;
using tvm::neg;
using tvm::not_equal;
using tvm::pow;
using tvm::reinterpret;
using tvm::ret;
using tvm::right_shift;
using tvm::shapediv;
using tvm::sub;
using tvm::truncdiv;
using tvm::truncmod;

using tvm::isfinite;
using tvm::isinf;
using tvm::isnan;

using tvm::all;
using tvm::any;
using tvm::max;
using tvm::min;
using tvm::prod;
using tvm::sum;

using tvm::ceil;
using tvm::fast_erf_float_expr;
using tvm::floor;
using tvm::LargeUIntImm;
using tvm::nearbyint;
using tvm::q_multiply_shift;
using tvm::round;
using tvm::trunc;

using tvm::acos;
using tvm::acosh;
using tvm::asin;
using tvm::asinh;
using tvm::atan;
using tvm::atanh;
using tvm::clz;
using tvm::cos;
using tvm::cosh;
using tvm::erf;
using tvm::exp;
using tvm::exp10;
using tvm::exp2;
using tvm::log;
using tvm::log10;
using tvm::log1p;
using tvm::log2;
using tvm::popcount;
using tvm::rsqrt;
using tvm::sigmoid;
using tvm::sin;
using tvm::sinh;
using tvm::sqrt;
using tvm::tan;
using tvm::tanh;

using tvm::atan2;
using tvm::copysign;
using tvm::hypot;
using tvm::ldexp;
using tvm::nextafter;

using tvm::tir::IsPointerType;

using tvm::tir::const_false;
using tvm::tir::const_true;
using tvm::tir::make_const;
using tvm::tir::make_zero;

using tvm::tir::as_const_int;

using tvm::tir::is_const_int;
using tvm::tir::is_const_number;
using tvm::tir::is_no_op;
using tvm::tir::is_one;
using tvm::tir::is_zero;

using tvm::tir::foldl;

using tvm::tir::is_const_int;
using tvm::tir::is_const_number;
using tvm::tir::is_const_power_of_two_integer;
using tvm::tir::is_negative_const;
using tvm::tir::is_no_op;
using tvm::tir::is_positive_const;

using tvm::tir::make_const;
using tvm::tir::make_zero;
using tvm::tir::MakeConstScalar;

using tvm::Range;
using tvm::runtime::Array;
using tvm::tir::Buffer;
using tvm::tir::BufferLoad;
using tvm::tir::decl_buffer;
using tvm::tir::IterVar;
using tvm::tir::IterVarType;
using tvm::tir::Var;

void OpTest();

}  // namespace op_test
