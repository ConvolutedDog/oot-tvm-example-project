#include "ir/type-test.h"
#include "test-func-registry.h"

using tvm::Span;

using tvm::FuncType;
using tvm::PointerType;
using tvm::PrimType;
using tvm::TupleType;
using tvm::Type;

using tvm::runtime::Array;
using tvm::runtime::DataType;

namespace type_test {

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
