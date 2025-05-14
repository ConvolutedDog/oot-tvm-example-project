#include "node/reflection-test.h"
#include "test-func-registry.h"
#include "tvm/ir/expr.h"

namespace reflection_test {

std::ostream &operator<<(std::ostream &os, const tvm::runtime::NDArray &arr) {  // NOLINT
  os << "Not implemented.\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, std::vector<std::string> &vec) {
  for (auto &s : vec)
    os << s << " ";
  return os;
}

void AttrVisitorTest() {
  LOG_SPLIT_LINE("AttrVisitorTest");
  PrimExpr start = 4;
  PrimExpr end = 8;

  Range range{start, end};
  Var var{"x", DataType::Int(32)};
  IterVar itervar{range, var, IterVarType::kThreadIndex, "threadidx.x"};

  MyIRSerializer serializer;
  const_cast<IterVarNode *>(itervar.as<IterVarNode>())->VisitAttrs(&serializer);
}

void ReflectionVTableTest() {
  LOG_SPLIT_LINE("ReflectionVTableTest");
  ReflectionVTable *vtable = ReflectionVTable::Global();

  /// Test PrimExpr

  PrimExpr start = 4;
  std::vector<std::string> attrnames =
      vtable->ListAttrNames(const_cast<PrimExprNode *>(start.get()));
  LOG_PRINT_VAR(attrnames);

  MyIRSerializer serializer;
  vtable->VisitAttrs(const_cast<PrimExprNode *>(start.get()), &serializer);

  std::string reprbytes = "tmp";
  vtable->GetReprBytes(const_cast<PrimExprNode *>(start.get()), &reprbytes);
  LOG_PRINT_VAR(reprbytes);

  /// Test IterVar

  PrimExpr end = 8;
  Range range{start, end};
  Var var{"x", DataType::Int(32)};
  IterVar itervar{range, var, IterVarType::kThreadIndex, "threadidx.x"};

  attrnames = vtable->ListAttrNames(const_cast<IterVarNode *>(itervar.get()));
  LOG_PRINT_VAR(attrnames);
  Range retvalrange = vtable->GetAttr(const_cast<IterVarNode *>(itervar.get()), "dom");
  LOG_PRINT_VAR(retvalrange);

  vtable->VisitAttrs(const_cast<IterVarNode *>(itervar.get()), &serializer);

  vtable->GetReprBytes(const_cast<IterVarNode *>(itervar.get()), &reprbytes);
  LOG_PRINT_VAR(reprbytes);

  /// New object

  ObjectPtr<Object> x = vtable->CreateInitObject(itervar->_type_key, "tmpitervar");
  vtable->VisitAttrs(const_cast<IterVarNode *>(IterVar(x).get()), &serializer);
}

}  // namespace reflection_test

void AttrVisitorTest() { reflection_test::AttrVisitorTest(); }
void ReflectionVTableTest() { reflection_test::ReflectionVTableTest(); }

namespace {

REGISTER_TEST_SUITE(AttrVisitorTest);
REGISTER_TEST_SUITE(ReflectionVTableTest);

}  // namespace
