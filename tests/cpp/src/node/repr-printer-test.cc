#include "node/repr-printer-test.h"
#include "test-func-registry.h"
#include "tvm/runtime/memory.h"
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/stmt.h>

namespace repr_printer_test {

/// @brief The `tvm::Dump()` function just calls `std::cerr <<`, and the overrided
/// `operator<<` if located at `tvm/node/repr_printer.h`:
///     inline std::ostream& operator<<(std::ostream& os, const ObjectRef& n) {
///       ReprPrinter(os).Print(n);
///       return os;
///     }
/// Here, it uses `ReprPrinter` to print the `ObjectRef` instance. There is a static
/// `NodeFunctor` instance:
///     using FType = NodeFunctor<void(const ObjectRef&, ReprPrinter*)>;
///     TVM_DLL static FType& vtable() {
///       static FType inst;
///       return inst;
///     }
///
/// Please refer to `functor-test.h` for more details about `NodeFunctor`.
///
/// In fact, the static `FType` instance is used to dispatch to different print functions
/// based on the type of the input node. TVM uses the macro `TVM_STATIC_IR_FUNCTOR` to
/// register the print function for a node type. For example, the `OpNode`:
///     TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
///         .set_dispatch<OpNode>([](const ObjectRef& ref, ReprPrinter* p) {
///           auto* node = static_cast<const OpNode*>(ref.get());
///           p->stream << "Op(" << node->name << ")";
///         });
void DumpTest() {
  LOG_SPLIT_LINE("DumpTest");

  /// Test Dump PrimExpr
  PrimExpr p = 100;
  tvm::Dump(p);

  /// Test IRModule
  Var a{"a", DataType::Int(32, 1)};
  Var b{"b", DataType::Int(32, 1)};

  LetStmt let = LetStmt{
      a, 100, LetStmt{b, 200, Evaluate{a + b}}
  };
  LOG_SPLIT_LINE("LOG_PRINT_VAR(let)");
  LOG_PRINT_VAR(let);

  PrimFunc primfunc{
      {a, b},
      let
  };
  LOG_SPLIT_LINE("LOG_PRINT_VAR(primfunc)");
  LOG_PRINT_VAR(primfunc);

  IRModule mod = IRModule::FromExpr(primfunc);
  LOG_SPLIT_LINE("LOG_PRINT_VAR(mod)");
  LOG_PRINT_VAR(mod);

  Evaluate expr = Evaluate{200 > 100 ? a : b};
  PrimFunc exprfunc{
      {a, b},
      expr
  };
  mod->Add(GlobalVar{"v"}, exprfunc);

  LOG_SPLIT_LINE("tvm::Dump(mod)");
  tvm::Dump(mod);

  LOG_SPLIT_LINE("tvm::Dump(mod->())");
  tvm::Dump(mod.operator->());
}

class TestNode : public ::tvm::runtime::Object {
public:
  explicit TestNode(::tvm::runtime::String s) : teststring(std::move(s)) {
    type_index_ = RuntimeTypeIndex();
  }
  const ::tvm::runtime::String &GetTestString() const { return teststring; }
  static constexpr const uint32_t _type_index = ::tvm::runtime::TypeIndex::kDynamic;
  static constexpr const char *_type_key = "test.TestNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestNode, ::tvm::runtime::Object);

private:
  ::tvm::runtime::String teststring;
};

class Test : public ::tvm::runtime::ObjectRef {
public:
  explicit Test(const ::tvm::runtime::String &teststring) {
    data_ = ::tvm::runtime::make_object<TestNode>(teststring);
  }
  TVM_DEFINE_OBJECT_REF_METHODS(Test, ::tvm::runtime::ObjectRef, TestNode);
};

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TestNode>([](const tvm::runtime::ObjectRef &ref, ReprPrinter *p) {
      auto *node = static_cast<const TestNode *>(ref.get());
      p->stream << "This is my TestNode: ^^^(" << node->GetTestString() << ")^^^";
    });

void ReprPrinterTest() {
  LOG_SPLIT_LINE("ReprPrinterTest");

  Test test{"Hello World!"};
  LOG_PRINT_VAR(test);
}

void AsLegacyReprTest() {
  LOG_SPLIT_LINE("AsLegacyReprTest");

  PrimExpr p = 100;

  /// The early version of the TVM printing API was likely more oriented towards returning
  /// strings (for purposes like logging or file writing), but later optimizations to
  /// output dumps directly are more in line with common usage. APIs marked as 'Legacy'
  /// are usually interfaces reserved from older versions.
  ///
  /// Unlike `Dump()`, `AsLegacyRepr()` returns a string.
  std::string ps = AsLegacyRepr(p);
  LOG_PRINT_VAR(ps);
}

void ReprLegacyPrinterTest() {
  LOG_SPLIT_LINE("ReprLegacyPrinterTest");

  /// Similar to `ReprPrinterTest`.
  LOG_PRINT_VAR("Similar to `ReprPrinterTest`.");
}

}  // namespace repr_printer_test

void AsLegacyReprTest() { repr_printer_test::AsLegacyReprTest(); }
void ReprPrinterTest() { repr_printer_test::ReprPrinterTest(); }
void ReprLegacyPrinterTest() { repr_printer_test::ReprLegacyPrinterTest(); }
void DumpTest() { repr_printer_test::DumpTest(); }

namespace {

REGISTER_TEST_SUITE(AsLegacyReprTest);
REGISTER_TEST_SUITE(ReprPrinterTest);
REGISTER_TEST_SUITE(DumpTest);
REGISTER_TEST_SUITE(ReprLegacyPrinterTest);

}  // namespace
