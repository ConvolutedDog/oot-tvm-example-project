#include "node/serialization-test.h"
#include "test-func-registry.h"
#include <tvm/ir/module.h>
#include <tvm/runtime/data_type.h>

namespace serialization_test {

void NodeSerializationTest() {
  LOG_SPLIT_LINE("NodeSerializationTest");

  /// @brief Define a GlobalVar.
  GlobalVar globalvar("globalvar");

  /// Create tvm::relax::Function
  Expr opexpr = tvm::Op::Get("relax.nn.conv2d");
  Var arg1{"arg1", tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4}};
  Var arg2{"arg2", tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4}};
  Call call{
      opexpr, {arg1, arg2}
  };
  Function func{
      {arg1, arg2},
      call,
      tvm::relax::TensorStructInfo{tvm::DataType::Float(32), 4},
      true,
  };

  /// @note TVMScript cannot print functions of type: BaseFunc
  IRModule irmodule2{{std::pair<GlobalVar, BaseFunc>{globalvar, func}}};
  LOG_PRINT_VAR(irmodule2);

  /// SaveJSON
  LOG_SPLIT_LINE("SaveJSON");
  std::string jsonoutput = SaveJSON(irmodule2);
  LOG_PRINT_VAR(jsonoutput);

  /// LoadJSON
  LOG_SPLIT_LINE("LoadJSON");
  IRModule irmodule3 = LoadJSON(jsonoutput).as<IRModule>().value();
  LOG_PRINT_VAR(irmodule3);

  LOG_PRINT_VAR(SaveJSON(irmodule3) == jsonoutput);
}

}  // namespace serialization_test

REGISTER_TEST_SUITE(serialization_test::NodeSerializationTest,
                    node_serialization_test_NodeSerializationTest);
