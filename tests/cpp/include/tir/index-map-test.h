#include "tvm/tir/index_map.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>

namespace index_map_test {

using tvm::tir::IndexMap;
using tvm::tir::Substitute;

using tvm::DataType;
using tvm::PrimExpr;
using tvm::Range;
using tvm::runtime::Array;
using tvm::runtime::NDArray;
using tvm::runtime::String;
using tvm::tir::Var;

void TirIndexMapTest();
void TirSubstituteTest();

}  // namespace index_map_test
