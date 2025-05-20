#include "ir/expr-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>

using std::string;

using tvm::PrimExpr;
using tvm::runtime::DataType;
using tvm::runtime::operator<<;

using tvm::Bool;
using tvm::FloatImmNode;
using tvm::Integer;
using tvm::IntImmNode;
using tvm::Range;

#define PRINT_TOP_LINE(hint)                                                             \
  LOG_SPLIT_LINE(string(" Constant fold for Op ") + string(#hint) + string(" "));

#define PRINT_VARS3(a, b, c)                                                             \
  LOG_PRINT_VAR(a)                                                                       \
  LOG_PRINT_VAR(b)                                                                       \
  LOG_PRINT_VAR(c)

#define PRINT_VARS2(a, b)                                                                \
  LOG_PRINT_VAR(a)                                                                       \
  LOG_PRINT_VAR(b)

#define GET_DLDATATYPE(primExprA, primExprB)                                             \
  auto dldtypeA = DLDataType((primExprA).dtype());                                       \
  auto dldtypeB = DLDataType((primExprB).dtype());

#define ASSERT_INT_DLDATATYPE(dldatatype)                                                \
  assert((dldatatype).code == DLDataTypeCode::kDLInt && "error datatype!")

#define ASSERT_FLOAT_DLDATATYPE(dldatatype)                                              \
  assert((dldatatype).code == DLDataTypeCode::kDLFloat && "error datatype!")

#define ASSERT_BOOL_DLDATATYPE(primExpr)                                                 \
  assert((primExpr).get()->dtype.is_bool() && "error datatype!")

#define CAST_TO_NODE(primExpr, res, NodeType)                                            \
  const NodeType *res = (primExpr).as<NodeType>()

#define TEST_OPERATOR_INT_FLOAT(primExprA, primExprB, op, hint)                          \
  {                                                                                      \
    PRINT_TOP_LINE(hint)                                                                 \
    GET_DLDATATYPE(primExprA, primExprB)                                                 \
    ASSERT_INT_DLDATATYPE(dldtypeA);                                                     \
    ASSERT_FLOAT_DLDATATYPE(dldtypeB);                                                   \
    PrimExpr primExprOp = primExprA op primExprB;                                        \
    CAST_TO_NODE(primExprA, pa, IntImmNode);                                             \
    CAST_TO_NODE(primExprB, pb, FloatImmNode);                                           \
    CAST_TO_NODE(primExprOp, popres, FloatImmNode);                                      \
    PRINT_VARS3(pa->value, pb->value, popres->value)                                     \
  }

#define TEST_OPERATOR_INT_INT(primExprA, primExprB, op, hint)                            \
  {                                                                                      \
    PRINT_TOP_LINE(hint)                                                                 \
    GET_DLDATATYPE(primExprA, primExprB)                                                 \
    ASSERT_INT_DLDATATYPE(dldtypeA);                                                     \
    ASSERT_INT_DLDATATYPE(dldtypeB);                                                     \
    PrimExpr primExprOp = primExprA op primExprB;                                        \
    CAST_TO_NODE(primExprA, pa, IntImmNode);                                             \
    CAST_TO_NODE(primExprB, pb, IntImmNode);                                             \
    CAST_TO_NODE(primExprOp, popres, IntImmNode);                                        \
    PRINT_VARS3(pa->value, pb->value, popres->value)                                     \
  }

#define TEST_OPERATOR_INT(primExprA, op, hint)                                           \
  {                                                                                      \
    PRINT_TOP_LINE(hint)                                                                 \
    GET_DLDATATYPE(primExprA, primExprA)                                                 \
    ASSERT_INT_DLDATATYPE(dldtypeA);                                                     \
    PrimExpr primExprOp = op primExprA;                                                  \
    CAST_TO_NODE(primExprA, pa, IntImmNode);                                             \
    CAST_TO_NODE(primExprOp, popres, IntImmNode);                                        \
    PRINT_VARS2(pa->value, popres->value)                                                \
  }

#define TEST_OPERATOR_BOOL_BOOL(primExprA, primExprB, op, hint)                          \
  {                                                                                      \
    PRINT_TOP_LINE(hint)                                                                 \
    ASSERT_BOOL_DLDATATYPE(primExprA);                                                   \
    ASSERT_BOOL_DLDATATYPE(primExprB);                                                   \
    PrimExpr primExprOp = primExprA op primExprB;                                        \
    CAST_TO_NODE(primExprA, pa, IntImmNode);                                             \
    CAST_TO_NODE(primExprB, pb, IntImmNode);                                             \
    CAST_TO_NODE(primExprOp, popres, IntImmNode);                                        \
    PRINT_VARS3(pa->value, pb->value, popres->value)                                     \
  }

#define TEST_OPERATOR_BOOL(primExprA, op, hint)                                          \
  {                                                                                      \
    PRINT_TOP_LINE(hint)                                                                 \
    ASSERT_BOOL_DLDATATYPE(primExprA);                                                   \
    PrimExpr primExprOp = op primExprA;                                                  \
    CAST_TO_NODE(primExprA, pa, IntImmNode);                                             \
    CAST_TO_NODE(primExprOp, popres, IntImmNode);                                        \
    PRINT_VARS2(pa->value, popres->value)                                                \
  }

namespace expr_test {

void IrPrimExprTest() {
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
  // LOG_PRINT_VAR(DLDataTypeCode::kDLBool);

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
  /// @note can't cast `~primeExprA` into IntImmNode directly
  /// `(~primExprA).as<IntImmNode>()` return the nullptr
  /// error:  TEST_OPERATOR_INT(primExprA, ~, bitwise neg);
  PrimExpr bitwiseA = ~primExprA;
  std::cout << bitwiseA << std::endl;
  std::cout << bitwiseA->GetTypeKey() << std::endl;//tir.Call

  TEST_OPERATOR_INT_INT(primExprA, primExprC, &, bitwise and);
  TEST_OPERATOR_INT_INT(primExprA, primExprC, |, bitwise or);

  /// Constant fold for BOOL Nodes.
  TEST_OPERATOR_BOOL(primExprD, !, not);
  TEST_OPERATOR_BOOL_BOOL(primExprD, primExprE, &&, and);
  TEST_OPERATOR_BOOL_BOOL(primExprD, primExprE, ||, or);
}

void IrBoolTest() {
  LOG_SPLIT_LINE("IrBoolTest");
  auto b = Bool(true);
  LOG_PRINT_VAR(b);
  LOG_PRINT_VAR(!b);
  LOG_PRINT_VAR((bool)b);
  LOG_PRINT_VAR(b.dtype());

  auto a = Bool(false);
  LOG_PRINT_VAR(b && false);
  LOG_PRINT_VAR(b && a);
  LOG_PRINT_VAR(b || false);
  LOG_PRINT_VAR(b || a);
  LOG_PRINT_VAR(b == false);
  LOG_PRINT_VAR(b == a);
}

void IrIntegerTest() {
  LOG_SPLIT_LINE("IrIntegerTest");
  Integer a = Integer(23);
  LOG_PRINT_VAR(a.IntValue());
  enum class X { aa = 20, bb = 21, cc = 22, SIZE = 23 };
  Integer b = Integer(X::SIZE);
  LOG_PRINT_VAR(b.IntValue());
  LOG_PRINT_VAR(b == a);
}

void IrRangeTest() {
  LOG_SPLIT_LINE("IrRangeTest");
  PrimExpr primExprA = 4;
  PrimExpr primExprB = 5;

  PrimExpr primExprC = 4;

  Range r = Range(primExprA * primExprB, primExprA * 2 * primExprB + primExprC);
  auto rnode = *(r.get());
  LOG_PRINT_VAR(rnode.min);
  LOG_PRINT_VAR(rnode.extent);
  LOG_PRINT_VAR(primExprA * 2 * primExprB + primExprC - primExprA * primExprB);
}

}  // namespace expr_test

REGISTER_TEST_SUITE(expr_test::IrPrimExprTest, ir_expr_test_IrPrimExprTest);
REGISTER_TEST_SUITE(expr_test::IrRangeTest, ir_expr_test_IrRangeTest);
REGISTER_TEST_SUITE(expr_test::IrIntegerTest, ir_expr_test_IrIntegerTest);
REGISTER_TEST_SUITE(expr_test::IrBoolTest, ir_expr_test_IrBoolTest);
