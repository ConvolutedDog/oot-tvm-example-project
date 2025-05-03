#include "ir/type-test.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n'
#define LOG_SPLIT_LINE(stmt) std::cout << "==============" << (stmt) << "==============\n"

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

}  // namespace type_test

void PrimTypeTest() {
  LOG_SPLIT_LINE("PrimTypeTest");
  DLDataType dldtype{DLDataTypeCode::kDLFloat, 32, 1};
  DataType dtype{dldtype};
  PrimType primetype{dtype};
  LOG_PRINT_VAR(type_test::DataType2Str(primetype.get()->dtype));
  LOG_PRINT_VAR(primetype.get()->span);
}

void PointerTypeTest() {
  LOG_SPLIT_LINE("PointerTypeTest");

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

void TupleTypeTest() {
  LOG_SPLIT_LINE("TupleTypeTest");

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

void FuncTypeTest() {
  LOG_SPLIT_LINE("FuncTypeTest");

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
