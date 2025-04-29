#include "../include/expr-test.h"

using std::string;

using tvm::PrimExpr;
using tvm::runtime::DataType;
using tvm::runtime::operator<<;

using tvm::Bool;
using tvm::FloatImmNode;
using tvm::IntImmNode;

namespace expr_test {}

#define PRINT_TOP_LINE(hint)                                                             \
  LOG_SPLIT_LINE(string(" Constant fold for Op ") + string(#hint) + string(" "));

#define PRINT_VARS3(a, b, c)                                                             \
  LOG_PRINT_VAR(a);                                                                      \
  LOG_PRINT_VAR(b);                                                                      \
  LOG_PRINT_VAR(c);

#define PRINT_VARS2(a, b)                                                                \
  LOG_PRINT_VAR(a);                                                                      \
  LOG_PRINT_VAR(b);

#define GET_DLDATATYPE(primExprA, primExprB)                                             \
  auto dldtypeA = DLDataType((primExprA).dtype());                                       \
  auto dldtypeB = DLDataType((primExprB).dtype());

#define ASSERT_INT_DLDATATYPE(dldatatype)                                                \
  assert((dldatatype).code == DLDataTypeCode::kDLInt && "error datatype!");

#define ASSERT_FLOAT_DLDATATYPE(dldatatype)                                              \
  assert((dldatatype).code == DLDataTypeCode::kDLFloat && "error datatype!");

#define ASSERT_BOOL_DLDATATYPE(primExpr)                                                 \
  assert((primExpr).get()->dtype.is_bool() && "error datatype!");

#define CAST_TO_NODE(primExpr, res, NodeType)                                            \
  const NodeType *res = (primExpr).as<NodeType>();

#define TEST_OPERATOR_INT_FLOAT(primExprA, primExprB, op, hint)                          \
  {                                                                                      \
    PRINT_TOP_LINE(hint);                                                                \
    GET_DLDATATYPE(primExprA, primExprB);                                                \
    ASSERT_INT_DLDATATYPE(dldtypeA);                                                     \
    ASSERT_FLOAT_DLDATATYPE(dldtypeB);                                                   \
    PrimExpr primExprOp = primExprA op primExprB;                                        \
    CAST_TO_NODE(primExprA, pa, IntImmNode)                                              \
    CAST_TO_NODE(primExprB, pb, FloatImmNode)                                            \
    CAST_TO_NODE(primExprOp, popres, FloatImmNode)                                       \
    PRINT_VARS3(pa->value, pb->value, popres->value);                                    \
  }

#define TEST_OPERATOR_INT_INT(primExprA, primExprB, op, hint)                            \
  {                                                                                      \
    PRINT_TOP_LINE(hint);                                                                \
    GET_DLDATATYPE(primExprA, primExprB);                                                \
    ASSERT_INT_DLDATATYPE(dldtypeA);                                                     \
    ASSERT_INT_DLDATATYPE(dldtypeB);                                                     \
    PrimExpr primExprOp = primExprA op primExprB;                                        \
    CAST_TO_NODE(primExprA, pa, IntImmNode)                                              \
    CAST_TO_NODE(primExprB, pb, IntImmNode)                                              \
    CAST_TO_NODE(primExprOp, popres, IntImmNode)                                         \
    PRINT_VARS3(pa->value, pb->value, popres->value);                                    \
  }

#define TEST_OPERATOR_INT(primExprA, op, hint)                                           \
  {                                                                                      \
    PRINT_TOP_LINE(hint);                                                                \
    GET_DLDATATYPE(primExprA, primExprA);                                                \
    ASSERT_INT_DLDATATYPE(dldtypeA);                                                     \
    PrimExpr primExprOp = op primExprA;                                                  \
    CAST_TO_NODE(primExprA, pa, IntImmNode)                                              \
    CAST_TO_NODE(primExprOp, popres, IntImmNode)                                         \
    PRINT_VARS2(pa->value, popres->value);                                               \
  }

#define TEST_OPERATOR_BOOL_BOOL(primExprA, primExprB, op, hint)                          \
  {                                                                                      \
    PRINT_TOP_LINE(hint);                                                                \
    ASSERT_BOOL_DLDATATYPE(primExprA);                                                   \
    ASSERT_BOOL_DLDATATYPE(primExprB);                                                   \
    PrimExpr primExprOp = primExprA op primExprB;                                        \
    CAST_TO_NODE(primExprA, pa, IntImmNode)                                              \
    CAST_TO_NODE(primExprB, pb, IntImmNode)                                              \
    CAST_TO_NODE(primExprOp, popres, IntImmNode)                                         \
    PRINT_VARS3(pa->value, pb->value, popres->value);                                    \
  }

#define TEST_OPERATOR_BOOL(primExprA, op, hint)                                          \
  {                                                                                      \
    PRINT_TOP_LINE(hint);                                                                \
    ASSERT_BOOL_DLDATATYPE(primExprA);                                                   \
    PrimExpr primExprOp = op primExprA;                                                  \
    CAST_TO_NODE(primExprA, pa, IntImmNode)                                              \
    CAST_TO_NODE(primExprOp, popres, IntImmNode)                                         \
    PRINT_VARS2(pa->value, popres->value);                                               \
  }

void PrimExprTest() {
  PrimExpr primExprA = 4;
  DataType dtypeA = primExprA.dtype();
  DLDataType dldtypeA = DLDataType(dtypeA);
  LOG_PRINT_VAR(dldtypeA);

  PrimExpr primExprB = 2.f;
  DataType dtypeB = primExprB.dtype();
  DLDataType dldtypeB = DLDataType(dtypeB);
  LOG_PRINT_VAR(dldtypeB);

  PrimExpr primExprC = 2;
  DataType dtypeC = primExprC.dtype();
  DLDataType dldtypeC = DLDataType(dtypeC);
  LOG_PRINT_VAR(dldtypeC);

  PrimExpr primExprD = Bool(false);
  DataType dtypeD = primExprD.dtype();
  DLDataType dldtypeD = DLDataType(dtypeD);
  LOG_PRINT_VAR(dldtypeD.code);
  LOG_PRINT_VAR(DLDataTypeCode::kDLBool);

  PrimExpr primExprE = Bool(true);

  /// Constant fold for INT-FLOAT Nodes.
  TEST_OPERATOR_INT_FLOAT(primExprA, primExprB, +, add);
  TEST_OPERATOR_INT_FLOAT(primExprA, primExprB, -, sub);
  TEST_OPERATOR_INT_FLOAT(primExprA, primExprB, *, mul);
  TEST_OPERATOR_INT_FLOAT(primExprA, primExprB, /, div);

  /// Constant fold for INT-INT Nodes.
  TEST_OPERATOR_INT_INT(primExprA, primExprC, +, add);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, -, sub);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, *, mul);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, /, div);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, <<, lsh);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, >>, rsh);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, >, gt);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, >=, ge);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, <, lt);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, <=, le);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, ==, eq);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, !=, ne);
  TEST_OPERATOR_INT(primExprA, -, neg);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, &, bitwise and);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, |, bitwise or);
  // TEST_OPERATOR_INT_INT(primExprA, primExprC, ~, bitwise neg); // Error

  /// Constant fold for BOOL Nodes.
  TEST_OPERATOR_BOOL(primExprD, !, not);
  TEST_OPERATOR_BOOL_BOOL(primExprD, primExprE, &&, and);
  TEST_OPERATOR_BOOL_BOOL(primExprD, primExprE, ||, or);
}
