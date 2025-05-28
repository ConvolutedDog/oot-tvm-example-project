#include "tvm/arith/int_set.h"

namespace int_set_test {

using tvm::arith::AsIntSet;
using tvm::arith::ConvertDomMap;
using tvm::arith::EstimateRegionLowerBound;
using tvm::arith::EstimateRegionStrictBound;
using tvm::arith::EstimateRegionUpperBound;
using tvm::arith::EvalSet;
using tvm::arith::EvalSetForEachSubExpr;
using tvm::arith::Intersect;
using tvm::arith::IntSet;
using tvm::arith::IntSetNode;
using tvm::arith::SignType;
using tvm::arith::Union;
using tvm::arith::UnionLowerBound;
using tvm::arith::UnionRegion;
using tvm::arith::UnionRegionLowerBound;

using tvm::tir::IterVar;
using tvm::tir::Var;
using tvm::tir::VarNode;

void ArithIntSetTest();

}  // namespace int_set_test
