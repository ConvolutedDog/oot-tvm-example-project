#include "ir/type-test.h"
#include "test-func-registry.h"
#include "tvm/ir/expr.h"
#include <tvm/runtime/logging.h>

using tvm::Span;

using tvm::FuncType;
using tvm::PointerType;
using tvm::PrimType;
using tvm::TupleType;
using tvm::Type;

using tvm::runtime::Array;
using tvm::runtime::DataType;

namespace type_test {

/// @brief Some explanation about DataType's vscale factor.
/// `DataType` can be used to store fixed-length vector or scalable vector. For the
/// fixed-length vectors, `lanes` should satisfy `static_cast<int16_t>(lanes) > 1`.
/// Fixed-length vectors' lanes can be accessed during the compile time. For the
/// scalable vectors, `lanes` should satisfy `static_cast<int16_t>(lanes) < -1`.
/// scalabe vectors' lanes can not be accessed during the compile time, but the
/// scalable vector length is determined by the hardware.
///
/// When we set the lanes of a fixed-length vector, `lanes = 4` only means that
/// "this operation semantically processes 4 elements" (the number of logical
/// lanes), and does not directly represent the actual vector length of the
/// hardware. For scalable vectors, we set `lanes = -4`, which means that the
/// number of real lanes of the scalable vector equals to the number of lanes
/// of the hardware basic vector unit * 4.
///
/// So, if we want to broadcast a scalr value to a scalable vector, we need to
/// call the `builtin::vscale` function to get the target's number of lanes
/// of the hardware basic vector unit. For example, in the operator+ function,
/// If op_a is a scalar value and op_b is a scalable vector, we need to broadcast
/// op_a to a scalable vector:
///   DataType dtype_a = op_a.dtype();  // lanes = 1
///   DataType dtype_b = op_b.dtype();  // lanes < -1
///   tir::Broadcast(op_a, tir::Mul(dtype_b.vscale_factor(),
///                  Call(DataType::Int(32), builtin::vscale(), {})));

std::string DataType2Str(const DataType &dtype) {
  return tvm::runtime::DLDataType2String((DLDataType)dtype);
}

/// Tests primitive type handling in TVM, demonstrating the relationship between:
/// - DLDataType: Low-level type descriptor (C-compatible POD struct) for cross-
///               language exchange.
/// - DataType: TVM's high-level type abstraction with extended functionality.
///
/// The conversion flow: DLDataType -> DataType -> PrimType shows how TVM bridges
/// between external interface types and internal type system representations.
///
/// Key points:
/// 1. DLDataType (bits/code/lanes) is minimal for framework interoperability.
/// 2. DataType wraps DLDataType with type system operations.
/// 3. PrimType represents final TVM IR type with source location (span).
void IrPrimTypeTest() {
  LOG_SPLIT_LINE("IrPrimTypeTest");
  DLDataType dldtype{DLDataTypeCode::kDLFloat, 32, 1};
  DataType dtype{dldtype};
  PrimType primetype{dtype};
  LOG_PRINT_VAR(type_test::DataType2Str(primetype.get()->dtype));
  LOG_PRINT_VAR(primetype.get()->span);
}

void IrPointerTypeTest() {
  LOG_SPLIT_LINE("IrPointerTypeTest");

  /// Create some types start
  DLDataType dldtype{DLDataTypeCode::kDLFloat, 32, 1};
  DataType dtype{dldtype};
  PrimType primetype{dtype};
  /// Create some types end

  PointerType ptype{primetype, "primtypesscope"};
  PointerType pptype{ptype, "ptypescope"};

  LOG_PRINT_VAR(ptype.get()->element_type);
  LOG_PRINT_VAR(ptype.get()->storage_scope);
  LOG_PRINT_VAR(ptype.get()->span);

  LOG_PRINT_VAR(pptype.get()->element_type);
  LOG_PRINT_VAR(pptype.get()->storage_scope);
  LOG_PRINT_VAR(pptype.get()->span);
}

void IrTupleTypeTest() {
  LOG_SPLIT_LINE("IrTupleTypeTest");

  /// Create some types start
  DLDataType dldtype{DLDataTypeCode::kDLFloat, 32, 1};
  DataType dtype{dldtype};
  PrimType primetype{dtype};

  PointerType ptype{primetype, "primtypesscope"};
  PointerType pptype{ptype, "ptypescope"};
  /// Create some types end

  std::initializer_list<Type> list{primetype, ptype, pptype};
  Array<Type> arr{list};

  TupleType tupletype{arr, Span()};
  LOG_PRINT_VAR(tupletype.get()->fields);
  LOG_PRINT_VAR(tupletype.Empty());

  std::initializer_list<Type> list2{primetype, ptype, pptype, tupletype};
  Array<Type> arr2{list2};
  TupleType tupletype2{arr2, Span()};
  LOG_PRINT_VAR(tupletype2.get()->fields);
}

void IrFuncTypeTest() {
  LOG_SPLIT_LINE("IrFuncTypeTest");

  /// Create some types start
  DLDataType dldtype{DLDataTypeCode::kDLFloat, 32, 1};
  DataType dtype{dldtype};
  PrimType primetype{dtype};

  PointerType ptype{primetype, "primtypesscope"};
  PointerType pptype{ptype, "ptypescope"};

  std::initializer_list<Type> list{primetype, ptype, pptype};
  Array<Type> arr{list};

  TupleType tupletype{arr, Span()};
  /// Create some types end

  std::initializer_list<Type> list2{primetype, ptype, pptype, tupletype};
  Array<Type> arr2{list2};
  FuncType functype{arr2, primetype};

  LOG_PRINT_VAR(functype.get()->arg_types);
  LOG_PRINT_VAR(functype.get()->ret_type);
}

}  // namespace type_test

REGISTER_TEST_SUITE(type_test::IrPrimTypeTest, ir_type_test_IrPrimTypeTest);
REGISTER_TEST_SUITE(type_test::IrPointerTypeTest, ir_type_test_IrPointerTypeTest);
REGISTER_TEST_SUITE(type_test::IrTupleTypeTest, ir_type_test_IrTupleTypeTest);
REGISTER_TEST_SUITE(type_test::IrFuncTypeTest, ir_type_test_IrFuncTypeTest);
