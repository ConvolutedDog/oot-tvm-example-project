#include "ir/function-test.h"
#include "test-func-registry.h"
#include "tvm/target/target.h"

namespace function_test {

std::ostream &operator<<(std::ostream &os, const LinkageType &linkage) {
  std::string ret;
  switch (linkage) {
    case LinkageType::kInternal: ret = "LinkageType::kInternal"; break;
    case LinkageType::kExternal: ret = "LinkageType::kExternal"; break;
    default: ret = "Unknown LinkageType";
  }
  os << ret;
  return os;
}

std::ostream &operator<<(std::ostream &os, const CallingConv &conv) {
  std::string ret;
  switch (conv) {
    case CallingConv::kDeviceKernelLaunch:
      ret = "CallingConv::kDeviceKernelLaunch";
      break;
    case CallingConv::kCPackedFunc: ret = "CallingConv::kCPackedFunc"; break;
    case CallingConv::kDefault: ret = "CallingConv::kDefault"; break;
    default: ret = "Unknown CallingConv";
  }
  os << ret;
  return os;
}

/// @brief Classes derived from tvm::BaseFunc can have attributes, which are stored in
/// `DictAttrs attrs`.
///                             |--------- tvm::relax::Function
///                             v
/// tvm::tir::PrimFunc -> tvm::BaseFunc <- tvm::relax::ExternFunc
void IrBaseFuncTest() {
  LOG_SPLIT_LINE("IrBaseFuncTest");
  ObjectPtr<BaseFuncNode> basefuncnodeptr = make_object<BaseFuncNode>();
  std::initializer_list<std::pair<String, ObjectRef>> init = {
      {"attr1",       String("attr1")                        },
      {"attr2",       tvm::PrimExpr(0)                       },
      {"attr3",       String("attr3")                        },
      /// Setting linkage type follows these rules:​​
      /// 1. ​​LinkageType::kExternal is set ONLY when:​​
      ///   - The kGlobalSymbol key is explicitly set ​​AND​​
      ///   - Its value is ​​not​​ NullValue<ObjectRef>().
      /// 2. ​​LinkageType::kInternal is set in ALL OTHER CASES (default):​​
      ///   - When kGlobalSymbol is ​​not set​​, ​​OR​​
      ///   - When kGlobalSymbol is set but its value ​​is​​
      ///     `NullValue<ObjectRef>()`.
      {kGlobalSymbol, NullValue<ObjectRef>()                 },
      {kTarget,       tvm::Target::Current()                 },
      {kCallingConv,  tvm::Integer{CallingConv::kCPackedFunc}}
  };
  Map<String, ObjectRef> map{init};
  DictAttrs dict(map);
  basefuncnodeptr->attrs = dict;

  BaseFunc basefunc(basefuncnodeptr);

  auto *basefuncnode = basefunc.as<BaseFuncNode>();

  LOG_PRINT_VAR(basefuncnode->attrs);
  /// Output:
  ///   basefuncnode->attrs: {"attr1": "attr1",
  ///                         "attr2": 0,
  ///                         "attr3": "attr3",
  ///                         "calling_conv": 0,
  ///                         "global_symbol": None,
  ///                         "target": None}

  LOG_PRINT_VAR(basefuncnode->GetAttr<String>("attr1"));
  /// Output: basefuncnode->GetAttr<String>("attr1"): "attr1"
  LOG_PRINT_VAR(basefuncnode->GetAttr<tvm::PrimExpr>("attr2"));
  /// Output: basefuncnode->GetAttr<tvm::PrimExpr>("attr2"): 0
  LOG_PRINT_VAR(basefuncnode->GetAttr<String>("attr3"));
  /// Output: basefuncnode->GetAttr<String>("attr3"): "attr3"
  LOG_PRINT_VAR(basefuncnode->GetAttr<tvm::Integer>(kCallingConv));
  /// Output: basefuncnode->GetAttr<tvm::Integer>(kCallingConv): 1
  LOG_PRINT_VAR(basefuncnode->GetAttr<tvm::Target>(kTarget));
  /// Output: basefuncnode->GetAttr<tvm::Target>(kTarget): (nullptr)
  LOG_PRINT_VAR(basefuncnode->GetAttr<String>(kGlobalSymbol));
  /// Output: basefuncnode->GetAttr<String>(kGlobalSymbol): (nullptr)

  LOG_PRINT_VAR(basefuncnode->HasNonzeroAttr("attr2"));
  /// Output: basefuncnode->HasNonzeroAttr("attr2"): 0
  LOG_PRINT_VAR(basefuncnode->HasNonzeroAttr(kCallingConv));
  /// Output: basefuncnode->HasNonzeroAttr(kCallingConv): 1

  LOG_PRINT_VAR(basefuncnode->GetLinkageType());
  /// Output: basefuncnode->GetLinkageType(): LinkageType::kInternal
}

}  // namespace function_test

REGISTER_TEST_SUITE(function_test::IrBaseFuncTest, ir_function_test_IrBaseFuncTest);
