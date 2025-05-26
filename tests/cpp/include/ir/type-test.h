#include "dlpack/dlpack.h"
#include "tvm/ir/source_map.h"
#include "tvm/ir/type.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/data_type.h"

/// @sa src/ir/type.cc

namespace type_test {

void IrPrimTypeTest();
void IrPointerTypeTest();
void IrTupleTypeTest();
void IrFuncTypeTest();

}  // namespace type_test
