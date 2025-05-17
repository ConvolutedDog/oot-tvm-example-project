#include "tir/expr-test.h"
#include "test-func-registry.h"
#include "tvm/relax/struct_info.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

namespace expr_test {

void TirExprTest() {
  LOG_SPLIT_LINE("ExprTest");

  /// StringImm
  StringImm stringimm{"Hello World!"};
  LOG_PRINT_VAR(stringimm);
  LOG_PRINT_VAR(stringimm->value);
  LOG_BLANK_LINE;

  /// Cast: Cast value from one data type to another.
  Cast cast{DataType::Float(32, 1), 123};
  LOG_PRINT_VAR(cast);  // T.Cast("float32", 123)
  LOG_BLANK_LINE;

  /// Add
  PrimExpr a{4};
  PrimExpr b{5};
  Add add{a, b};
  LOG_PRINT_VAR(add);  // T.Add(4, 5)
  LOG_PRINT_VAR(add->a);
  LOG_PRINT_VAR(add->b);
  LOG_BLANK_LINE;

  /// Sub/Mul/Div/Mod/FloorDiv/FloorMod/Min/Max
  Sub sub{a, b};
  Mul mul{a, b};
  Div div{a, b};
  Mod mod{a, b};
  FloorDiv floor_div{a, b};  // NOLINT
  FloorMod floor_mod{a, b};  // NOLINT
  Min min{a, b};
  Max max{a, b};
  LOG_PRINT_VAR(sub);        // T.Sub(4, 5)
  LOG_PRINT_VAR(mul);        // T.Mul(4, 5)
  LOG_PRINT_VAR(div);        // T.Div(4, 5)
  LOG_PRINT_VAR(mod);        // T.truncmod(4, 5)
  LOG_PRINT_VAR(floor_div);  // T.FloorDiv(4, 5)
  LOG_PRINT_VAR(floor_mod);  // T.FloorMod(4, 5)
  LOG_PRINT_VAR(min);        // T.min(4, 5)
  LOG_PRINT_VAR(max);        // T.max(4, 5)
  LOG_BLANK_LINE;

  // EQ/NE/LT/LE/GT/GE/And/Or/Not/Select
  EQ eq{a, b};
  NE ne{a, b};
  LT lt{a, b};
  LE le{a, b};
  GT gt{a, b};
  GE ge{a, b};
  tvm::Bool aa = tvm::Bool(true);
  tvm::Bool bb = tvm::Bool(false);
  And and_{aa, bb};  // NOLINT
  Or or_{aa, bb};    // NOLINT
  Not not_{aa};      // NOLINT
  Select select{tvm::Bool{false}, a, b};
  LOG_PRINT_VAR(eq);
  LOG_PRINT_VAR(ne);
  LOG_PRINT_VAR(lt);
  LOG_PRINT_VAR(le);
  LOG_PRINT_VAR(gt);
  LOG_PRINT_VAR(ge);
  LOG_PRINT_VAR(and_);
  LOG_PRINT_VAR(or_);
  LOG_PRINT_VAR(not_);
  LOG_PRINT_VAR(select);
  LOG_BLANK_LINE;
}

void BufferLoadTest() {
  LOG_SPLIT_LINE("BufferLoadTest");

  /// Define a Buffer instance.
  DataType dtype = DataType::Float(32, 4);
  Var data{
      "dataptr", PointerType{PrimType{dtype}, "global"}
  };
  Array<PrimExpr> shape{128, 128};
  Array<PrimExpr> strides{128, 1};
  PrimExpr elem_offset = PrimExpr(0);  // NOLINT
  String buffer_name{"buffer"};        // NOLINT
  int align = 64;
  int offset_factor = 64;           // NOLINT
  Array<IntImm> axis_separators{};  // NOLINT

  Buffer buffer = Buffer(data, dtype, shape, strides, elem_offset, buffer_name, align,
                         offset_factor, BufferType::kDefault, axis_separators, Span{});

  /// Define a BufferLoad instance.
  BufferLoad bufferload{
      buffer, {2,               4},
       Broadcast{tvm::Bool{true}, 4}
  };
  LOG_PRINT_VAR(bufferload);
}

void ProducerLoadTest() {
  /// Define a ProducerLoad instance.
  /// @todo (yangjianchao) Supplement more details about `ProducerLoad`.
  ProducerLoad producerload{
      Tensor{{2, 2},
             DataType::Float(32, 4),
             PlaceholderOp{"placeholder", {1, 2}, DataType::Float(32, 4)},
             1},
      {1, 1}
  };
  LOG_PRINT_VAR(producerload);
}

void RampTest() {
  /// Ramp: Construct a vector with lanes elements where its i-th element equals
  /// base + i * stride.  This is useful to construct a index for a continuous
  /// vector load.
  ///
  /// Examples:
  ///   - ramp(0, 1, 3) = [0, 1, 2]
  ///   - ramp(1, 2, 4) = [1, 3, 5, 7]
  {
    LOG_SPLIT_LINE("Ramp");
    PrimExpr base = 0, stride = 1, lanes = 3;
    Ramp ramp{base, stride, lanes};
    LOG_PRINT_VAR(ramp);
  }
}

void BroadcastTest() {
  /// Broadcast: Create a vector where all the elements are value.
  /// @sa buffer_test::BufferTest()
  {
    LOG_SPLIT_LINE("Broadcast");
    PrimExpr value = 1, lanes = 3;
    Broadcast broadcast{value, lanes};
    LOG_PRINT_VAR(broadcast);
  }
}

void LetTest() {
  /// Let: Let binding. Bind var to value then evaluate body.
  {
    LOG_SPLIT_LINE("Let");
    Var x{"x"};
    Let let{
        x, Add{x, 1},
         Add{x, 2}
    };
    LOG_PRINT_VAR(let);
  }
}

void TirCallTest() {
  /// Call
  ///
  /// \sa module_test::ModuleTest
  {
    LOG_SPLIT_LINE("Call");
    tvm::RelaxExpr opexpr = tvm::Op::Get("relax.nn.conv2d");
    Var arg1{"arg1"};
    Var arg2{"arg2"};
    Call call2{
        DataType::Float(32), opexpr, {arg1, arg2}
    };
    LOG_PRINT_VAR(call2);
  }
}

void ShuffleTest() {
  /// Shuffle instruction.
  ///   vec = concat(vectors)
  ///   result = (vec[indices[0]], vec[indices[1]] ...)
  {
    LOG_SPLIT_LINE("Shuffle");
    Array<PrimExpr> vectors{1, 2, 3, 4};
    Array<PrimExpr> indices{3, 2, 1, 0};
    Shuffle shuffle{vectors, indices};
    LOG_PRINT_VAR(shuffle);

    PrimExpr shuffleconcat{Shuffle::Concat(vectors)};
    LOG_PRINT_VAR(shuffleconcat);

    PrimExpr shuffleextractelement{Shuffle::ExtractElement(1, 0)};
    LOG_PRINT_VAR(shuffleextractelement);
  }
}

void CommReducerTest() {
  /// CommReducer: A commutative reducer node to represent a commutative binary
  /// operator with identity element.
  {
    LOG_SPLIT_LINE("CommReducer");
    Var x("x", DataType::Float(32)), y("y", DataType::Float(32));
    PrimExpr result = tvm::tir::Add(x, y);
    PrimExpr identity_element = tvm::tir::make_zero(x.dtype());  // NOLINT
    LOG_PRINT_VAR(identity_element);

    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }
}

void ReduceTest() {
  /// Reduce: Reduction operator
  {
    /// @todo (yangjianchao)
    LOG_SPLIT_LINE("Reduce");
  }
}

}  // namespace expr_test

REGISTER_TEST_SUITE(expr_test::TirExprTest, tir_expr_test_TirExprTest);
REGISTER_TEST_SUITE(expr_test::BufferLoadTest, tir_expr_test_BufferLoadTest);
REGISTER_TEST_SUITE(expr_test::ProducerLoadTest, tir_expr_test_ProducerLoadTest);
REGISTER_TEST_SUITE(expr_test::RampTest, tir_expr_test_RampTest);
REGISTER_TEST_SUITE(expr_test::BroadcastTest, tir_expr_test_BroadcastTest);
REGISTER_TEST_SUITE(expr_test::LetTest, tir_expr_test_LetTest);
REGISTER_TEST_SUITE(expr_test::TirCallTest, tir_expr_test_TirCallTest);
REGISTER_TEST_SUITE(expr_test::ShuffleTest, tir_expr_test_ShuffleTest);
REGISTER_TEST_SUITE(expr_test::CommReducerTest, tir_expr_test_CommReducerTest);
REGISTER_TEST_SUITE(expr_test::ReduceTest, tir_expr_test_ReduceTest);
