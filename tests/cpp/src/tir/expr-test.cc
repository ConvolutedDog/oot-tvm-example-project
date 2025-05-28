#include "tir/expr-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/data_type.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

namespace expr_test {

using ::tvm::tir::BufferNode;

class MyIRSerializer : public tvm::AttrVisitor {
  void Visit(const char *key, double *value) override {
    std::cout << " double:             " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, int64_t *value) override {
    std::cout << " int64_t:            " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, uint64_t *value) override {
    std::cout << " uint64_t:           " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, int *value) override {
    std::cout << " int:                " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, bool *value) override {
    std::cout << " bool:               " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, std::string *value) override {
    std::cout << " std::string:        " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, void **value) override {
    std::cout << " void:               " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, DataType *value) override {
    std::cout << " DataType:           " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, tvm::runtime::NDArray *value) override {
    std::cout << " runtime::NDArray:   " << key << "=" << *value << ";\n";
  }
  void Visit(const char *key, tvm::runtime::ObjectRef *value) override {
    std::cout << " runtime::ObjectRef: " << key << "=" << *value << ";\n";
  }
};

void TirExprTest() {
  LOG_SPLIT_LINE("ExprTest");

  /// StringImm
  StringImm stringimm{"Hello World!"};
  LOG_PRINT_VAR(stringimm);
  LOG_PRINT_VAR(stringimm->value);
  LOG_PRINT_VAR(stringimm->dtype);   // handle
  LOG_PRINT_VAR(stringimm.dtype());  // handle
  LOG_BLANK_LINE;

  /// Cast: Cast value from one data type to another.
  Cast cast{DataType::Float(32, 1), 123};
  LOG_PRINT_VAR(cast);  // T.Cast("float32", 123)
  LOG_PRINT_VAR(cast->dtype);
  LOG_PRINT_VAR(cast.dtype());
  LOG_PRINT_VAR(cast->value);
  LOG_BLANK_LINE;

  /// Add
  PrimExpr a{4};
  PrimExpr b{5};
  Add add{a, b};
  LOG_PRINT_VAR(add);  // T.Add(4, 5)
  LOG_PRINT_VAR(add->a);
  LOG_PRINT_VAR(add->b);
  LOG_PRINT_VAR(add->dtype);  // int32
  LOG_PRINT_VAR(add.dtype());
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

  LOG_PRINT_VAR(sub->dtype);        // int32
  LOG_PRINT_VAR(mul->dtype);        // int32
  LOG_PRINT_VAR(div->dtype);        // int32
  LOG_PRINT_VAR(mod->dtype);        // int32
  LOG_PRINT_VAR(floor_div->dtype);  // int32
  LOG_PRINT_VAR(floor_mod->dtype);  // int32
  LOG_PRINT_VAR(min->dtype);        // int32
  LOG_PRINT_VAR(max->dtype);        // int32
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
  /// @note Must use tvm::Bool params.
  And and_{aa, bb};  // NOLINT
  Or or_{aa, bb};    // NOLINT
  Not not_{aa};      // NOLINT
  Select select{tvm::Bool{false}, a, b};
  LOG_PRINT_VAR(eq);
  LOG_PRINT_VAR(eq->dtype);  // bool
  LOG_PRINT_VAR(ne);
  LOG_PRINT_VAR(ne->dtype);  // bool
  LOG_PRINT_VAR(lt);
  LOG_PRINT_VAR(lt->dtype);  // bool
  LOG_PRINT_VAR(le);
  LOG_PRINT_VAR(le->dtype);  // bool
  LOG_PRINT_VAR(gt);
  LOG_PRINT_VAR(gt->dtype);  // bool
  LOG_PRINT_VAR(ge);
  LOG_PRINT_VAR(ge->dtype);  // bool
  LOG_PRINT_VAR(and_);
  LOG_PRINT_VAR(and_->dtype);  // bool
  LOG_PRINT_VAR(or_);
  LOG_PRINT_VAR(or_->dtype);  // bool
  LOG_PRINT_VAR(not_);
  LOG_PRINT_VAR(not_->dtype);  // bool
  LOG_PRINT_VAR(select);
  LOG_PRINT_VAR(select->dtype);  // int32
  LOG_BLANK_LINE;
}

void TirBufferLoadTest() {
  LOG_SPLIT_LINE("BufferLoadTest");

  /// Define a Buffer instance. A Bufer's data type is equal to the type of elements it
  /// stored.
  DataType dtype = DataType::Float(32, 4);  // float32x4

  /// When binding the buffer to a variable, we need to specify the data type of the
  /// variable to be `tvm::PointerType` which points to `PrimType` elements.
  /// @todo (yangjianchao) Whether the pointered dtype of `Var` should be consistent with
  /// the type of the Buffer?
  Var data{
      "dataptr", PointerType{PrimType{dtype}, "global"}
  };
  /// This shape contains the shape as it is accessed by BufferLoad/BufferStore nodes, and
  /// used by the low-level code generators.
  Array<PrimExpr> shape{128, 128, 128};  // 2D 128 x 128 matrix buffer
  Array<PrimExpr> strides{};             // Row-major. Here, users can also don't specify
                                         // strides. Just use `strides{}` here and call
  // `buffer.MakeStrideView()` will generate strides
  // automatically (default to row-major: [128, 1]).
  /// The offset in terms of number of dtype elements (including lanes).
  PrimExpr elem_offset = PrimExpr(0);  // NOLINT
  String buffer_name{"buffer"};        // NOLINT
  /// The alignment of data in bytes.
  int align = 64;
  /// Factor of elem_offset field, elem_offset is guaranteed to be multiple of
  /// offset_factor. @ref https://www.zhihu.com/question/565420155
  int offset_factor = 64;  // NOLINT
  /// Axis separators is used to split the input axes into multiple sub-axes, which will
  /// be reflected in the output axes. The axis separators should be chosen from 0~n-1,
  /// where n is the number of dimensions of the buffer. The order of the axis separators
  /// should be in increasing order.
  /// @todo Supplement more details about axis_separators.
  // NOLINTNEXTLINE
  Array<IntImm> axis_separators{
      {IntImm{DataType::Int(32), 0}, IntImm{DataType::Int(32), 1}}
  };

  /// @brief BufferType:
  ///   /*! \brief buffer type */
  ///   enum BufferType : int {
  ///     kDefault = 1,
  ///     // Maps buffer[i][j][k] -> buffer[i][0][k] if dimension j's shape equals 1.
  ///     kAutoBroadcast = 2,
  ///   };
  ///
  /// 1. kDefault:
  ///    Normal buffer, no automatic broadcast. When accessing buffer[i][j][k], the data
  ///    is accessed strictly by the actual coordinates (i, j, k). If a dimension is
  ///    shape=1, you still need to explicitly specify an index (for example,
  ///    buffer[0][j][k]) when accessing that dimension. If it is not explicitly specified
  ///    as 0, it is out of bounds.
  /// 2. kAutoBroadcast:
  ///    Automatically broadcast axes with dimension 1. If a dimension is shape=1, it is
  ///    automatically broadcast when the dimension is accessed, i.e. buffer[i][j][k] is
  ///    mapped to buffer[0][j][k] (regardless of i).
  ///
  /// @ref src/script/ir_builder/tir/ir.cc
  /// @sa Buffer BufferDecl(...);
  Buffer buffer = Buffer(data, dtype, shape, strides, elem_offset, buffer_name, align,
                         offset_factor, BufferType::kDefault, axis_separators, Span{});

  LOG_PRINT_VAR(buffer);
  MyIRSerializer serializer;
  const_cast<BufferNode *>(buffer.get())->VisitAttrs(&serializer);

  /// GetFlattenedBuffer
  auto bufferflatten = buffer.GetFlattenedBuffer();
  LOG_PRINT_VAR(bufferflatten);
  const_cast<BufferNode *>(bufferflatten.get())->VisitAttrs(&serializer);
  LOG_PRINT_VAR(bufferflatten->data);
  LOG_PRINT_VAR(bufferflatten->shape);
  LOG_PRINT_VAR(bufferflatten->axis_separators);
  LOG_PRINT_VAR(bufferflatten->strides);
  LOG_PRINT_VAR(bufferflatten->elem_offset);
  LOG_PRINT_VAR(bufferflatten->name);
  LOG_PRINT_VAR(bufferflatten->data_alignment);
  LOG_PRINT_VAR(bufferflatten->offset_factor);
  LOG_PRINT_VAR(bufferflatten->buffer_type);

  /// Define a BufferLoad instance.
  // clang-format off
  /// Lanes of `predicate` of `Bufferload` must be consistent with the `dtype.lanes` of
  /// the `Buffer`. The shape of the indices must equal to the shape size of `Buffer`.
  /// The indices can set its last index to be a vector type (with DataType's lanes > 1).
  /// @todo (yangjianchao) Supplement more details about indices with its last index being
  /// with DataType's lanes > 1.
  BufferLoad bufferload{buffer, {2, 4,8}, Broadcast{tvm::Bool{true}, 4}};
  // clang-format on
  LOG_PRINT_VAR(bufferload);
  LOG_PRINT_VAR(bufferload->buffer->data);
  LOG_PRINT_VAR(bufferload->buffer->dtype);
  LOG_PRINT_VAR(bufferload->buffer->shape);
  LOG_PRINT_VAR(bufferload->buffer->axis_separators);
  LOG_PRINT_VAR(bufferload->buffer->strides);
  LOG_PRINT_VAR(bufferload->buffer->elem_offset);
  LOG_PRINT_VAR(bufferload->buffer->name);
  LOG_PRINT_VAR(bufferload->buffer->data_alignment);
  LOG_PRINT_VAR(bufferload->buffer->offset_factor);
  LOG_PRINT_VAR(bufferload->buffer->buffer_type);
  LOG_PRINT_VAR(bufferload->dtype);
  LOG_PRINT_VAR(bufferload->indices);
  LOG_PRINT_VAR(bufferload->predicate);
}

void TirProducerLoadTest() {
  /// Define a ProducerLoad instance.
  /// The information about `placeholder` is in tests/cpp/src/te/operation-test.cc.
  /// It generates a 16x3x224x224 `Tensor` which inherits from `DataProducer`.
  /// @todo There is no `indices` check in `ProducerLoad`'s constructor. May should check.
  ProducerLoad producerload{
      tvm::te::placeholder({16, 3, 224, 224},
      DataType::Float(32, 4), "placeholder"),
      {1,  1, 2,   3  }
  };
  LOG_PRINT_VAR(producerload);
  LOG_PRINT_VAR(producerload->producer);
  LOG_PRINT_VAR(producerload->indices);
  LOG_PRINT_VAR(producerload->dtype);  // Return Buffer type.
}

void TirRampTest() {
  /// Ramp: Construct a vector with `lanes` elements where its i-th element equals
  /// `base + i * stride`.  This is useful to construct a index for a continuous
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
    LOG_PRINT_VAR(ramp->base);
    LOG_PRINT_VAR(ramp->stride);
    LOG_PRINT_VAR(ramp->lanes);
    LOG_PRINT_VAR(ramp->dtype);
  }
}

void TirBroadcastTest() {
  /// Broadcast: Create a vector where all the elements are `value`.
  /// @sa buffer_test::TirBufferTest()
  {
    LOG_SPLIT_LINE("Broadcast");
    PrimExpr value = 1, lanes = 4;
    Broadcast broadcast{value, lanes};
    LOG_PRINT_VAR(broadcast);
    LOG_PRINT_VAR(broadcast->value);
    LOG_PRINT_VAR(broadcast->lanes);
    LOG_PRINT_VAR(broadcast->dtype);
  }
}

void TirLetTest() {
  /// Let: Let binding. Bind var to value then evaluate body.
  {
    LOG_SPLIT_LINE("Let");
    Var x{"x"};
    Let let{
        x, Add{x, 1}, // First let x = x + 1;
        Add{x, 2}  // Then return x + 2.
    };
    LOG_PRINT_VAR(let);
    LOG_PRINT_VAR(let->var);
    LOG_PRINT_VAR(let->value);
    LOG_PRINT_VAR(let->body);
    LOG_PRINT_VAR(let->dtype);
  }
}

void TirCallTest() {
  /// Call
  ///
  /// \sa module_test::IrModuleTest
  {
    LOG_SPLIT_LINE("Call");
    tvm::RelaxExpr opexpr = tvm::Op::Get("relax.nn.conv2d");
    Var arg1{"arg1"};
    Var arg2{"arg2"};
    Call call2{
        DataType::Float(32), opexpr, {arg1, arg2}
    };
    LOG_PRINT_VAR(call2);
    LOG_PRINT_VAR(call2->op);
    LOG_PRINT_VAR(call2->args);
    LOG_PRINT_VAR(call2->dtype);

    /// The `tir.Call` in python frontend can accept several variant params,
    /// please refer to `TVM_REGISTER_GLOBAL("tir.Call")` in tvm/src/tir/ir/expr.cc.
  }
}

/// @brief `Shuffle` is one of the most important operators in TVM. We have learned about
/// that `Broadcast` operator is use to expand a scalar to a vector (all the lanes are
/// same), and `Ramp` is used to expand a start value to a vector by strides (all the
/// lanes are different). And `Shuffle` gives user the ability to expand a vector in a
/// fine-grained way, for example, we can use `Shuffle` to concat values for each lane.
/// For example, we have 4 values for 4 lanes: {1.0f, 8.0f, 4.0f, 5.0f}, and we want
/// store them into a float32x4 buffer. Then we can use `Shuffle` to concat the values
/// vector together, and give it a index vector like {0,1,2,3} or {3,1,2,0} that tell
/// the values vector which lane to take the value from. And the result of the `Shuffle`
/// operator will be {1.0f, 8.0f, 4.0f, 5.0f} or {5.0f, 8.0f, 4.0f, 1.0f}.
void TirShuffleTest() {
  /// Shuffle instruction.
  ///   vec = concat(vectors)
  ///   result = (vec[indices[0]], vec[indices[1]] ...)
  {
    LOG_SPLIT_LINE("Shuffle");
    Array<PrimExpr> vectors{1, 2, 3, 4};
    Array<PrimExpr> indices{3, 2, 1, 0};
    Shuffle shuffle{vectors, indices};
    LOG_PRINT_VAR(shuffle);
    LOG_PRINT_VAR(shuffle->vectors);
    LOG_PRINT_VAR(shuffle->indices);
    LOG_PRINT_VAR(shuffle->dtype);

    auto var1 = tvm::tir::Var("var1", DataType::Int(32, 4));
    auto var2 = tvm::tir::Var("var2", DataType::Int(32, 4));

    Array<PrimExpr> vector2({var1, var2});
    PrimExpr shuffleconcat2{Shuffle::Concat(vector2)};
    LOG_PRINT_VAR(shuffleconcat2);

    PrimExpr shuffleconcat{Shuffle::Concat(vectors)};
    LOG_PRINT_VAR(shuffleconcat);

    auto x1 = tvm::tir::Broadcast{1, 4};
    auto x2 = tvm::tir::Broadcast{2, 4};
    auto x3 = tvm::tir::Broadcast{3, 4};
    auto x4 = tvm::tir::Broadcast{4, 4};
    Array<PrimExpr> withlanesvectors{x1, x2, x3, x4};
    PrimExpr shuffledvectors{Shuffle::Concat(withlanesvectors)};
    LOG_PRINT_VAR(shuffledvectors);

    LOG_PRINT_VAR(Shuffle::ExtractElement(shuffle, 0));
    PrimExpr shuffleextractelement{Shuffle::ExtractElement(shuffledvectors, 1)};
    LOG_PRINT_VAR(shuffleextractelement);
  }
}

void TirCommReducerTest() {
  /// CommReducer: A commutative reducer node to represent a commutative binary
  /// operator with identity element.
  {
    LOG_SPLIT_LINE("CommReducer");
    Var x("x", DataType::Float(32)), y("y", DataType::Float(32));
    PrimExpr result = tvm::tir::Add(x, y);

    // clang-format off
    /// `identity_element` is the identity element of the operator. For example:
    /// 1. sum: make_zero(source.dtype(), span); (sum_result + 0 = sum_result)
    /// 2. all: make_const(source.dtype(), true, span); (all(all_result, true) = all_result)
    /// 3. any: make_const(source.dtype(), false, span); (any(any_result, false) = any_result)
    /// 4. max: min_value(source.dtype(), span); (max(max_result, min_value) = max_result)
    /// 5. min: max_value(source.dtype(), span); (min(min_result, max_value) = min_result)
    /// 6. prod: make_const(source.dtype(), 1, span); (prod(prod_result, 1) = prod_result)
    // clang-format on
    PrimExpr identity_element = tvm::tir::make_zero(x.dtype());  // NOLINT
    LOG_PRINT_VAR(identity_element);

    /// @ref Refer to the implementation of `sum`:
    // clang-format off
    /// PrimExpr sum(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span
    /// span) {
    ///   Var x("x", source.dtype(), span), y("y", source.dtype(), span);
    ///   PrimExpr result = tir::Add(x, y, span);
    ///   PrimExpr identity_element = make_zero(source.dtype(), span);
    ///   tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result},
    ///   {identity_element}, span); return tir::Reduce(combiner, {source}, rdom,
    ///   make_const(DataType::Bool(1), true), 0, init, span);
    /// }
    // clang-format on

    /// `lhs`, `rhs`, `result` and `identity_element` must have the same size.
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
    LOG_PRINT_VAR(combiner->lhs);
    LOG_PRINT_VAR(combiner->rhs);
    LOG_PRINT_VAR(combiner->result);
    LOG_PRINT_VAR(combiner->identity_element);
  }

  /// sum
  LOG_SPLIT_LINE("sum");
  {
    DataType dtype = DataType::Float(32);
    Var x("x", dtype), y("y", dtype);
    PrimExpr result = tvm::tir::Add(x, y);
    PrimExpr identity_element = tvm::te::make_zero(dtype);  // NOLINT
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }

  /// all
  LOG_SPLIT_LINE("all");
  {
    DataType dtype = DataType::Bool(1);
    Var x("x", dtype), y("y", dtype);
    PrimExpr result = tvm::tir::And(x, y);
    PrimExpr identity_element = tvm::te::make_const(dtype, true);  // NOLINT
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }

  /// any
  LOG_SPLIT_LINE("any");
  {
    DataType dtype = DataType::Bool(1);
    Var x("x", dtype), y("y", dtype);
    PrimExpr result = tvm::tir::Or(x, y);
    PrimExpr identity_element = tvm::te::make_const(dtype, false);  // NOLINT
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }

  /// max
  LOG_SPLIT_LINE("max");
  {
    DataType dtype = DataType::Float(32);
    Var x("x", dtype), y("y", dtype);
    PrimExpr result = tvm::tir::Max(x, y);
    PrimExpr identity_element = tvm::min_value(dtype);  // NOLINT
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }

  /// min
  LOG_SPLIT_LINE("min");
  {
    DataType dtype = DataType::Float(32);
    Var x("x", dtype), y("y", dtype);
    PrimExpr result = tvm::tir::Min(x, y);
    PrimExpr identity_element = tvm::max_value(dtype);  // NOLINT
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }

  /// prod
  LOG_SPLIT_LINE("prod");
  {
    DataType dtype = DataType::Float(32);
    Var x("x", dtype), y("y", dtype);
    PrimExpr result = tvm::tir::Min(x, y);
    PrimExpr identity_element = tvm::te::make_const(dtype, 1);  // NOLINT
    CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
    LOG_PRINT_VAR(combiner);
  }
}

inline tvm::te::Tensor matmul(const tvm::te::Tensor &A,                          // NOLINT
                              const tvm::te::Tensor &B,                          // NOLINT
                              bool trans_a = false, bool trans_b = false,        // NOLINT
                              std::string name = "T_matmul",                     // NOLINT
                              std::string tag = "matmul") {                      // NOLINT
  tvm::Array<tvm::PrimExpr> output_shape{A->shape[trans_a ? 1 : 0],              // NOLINT
                                         B->shape[trans_b ? 0 : 1]};             // NOLINT
  auto k = tvm::te::reduce_axis(tvm::Range{0, A->shape[trans_a ? 0 : 1]}, "k");  // NOLINT
  auto l = [&](tvm::tir::Var i, tvm::tir::Var j) {                               // NOLINT
    return tvm::sum((trans_a ? A[k][i] : A[i][k]) *                              // NOLINT
                        (trans_b ? B[j][k] : B[k][j]),                           // NOLINT
                    {k});
  };
  return tvm::te::compute(output_shape, l, std::move(name), std::move(tag));
}

void TirReduceTest() {
  /// Reduce: Reduction operator
  {
    /// @ref Refer to the above matmul implementation.
    LOG_SPLIT_LINE("Reduce");

    Array<PrimExpr> shape{128, 128};
    DataType dtype = DataType::Float(32, 4);
    tvm::te::Tensor tensorA = tvm::te::placeholder(shape, dtype, "A");
    tvm::te::Tensor tensorB = tvm::te::placeholder(shape, dtype, "B");
    tvm::te::Tensor tensorC = matmul(tensorA, tensorB);
    LOG_PRINT_VAR(tensorC);

    tvm::tir::IterVar k{
        {0, tensorA->GetShape()[1]},
        tvm::te::var("k", DataType::Int(32)),
        tvm::tir::IterVarType::kCommReduce
    };
    // clang-format off
    std::function<PrimExpr(const Var &i, const Var &j)> func = [&](const Var &i,
                                                                   const Var &j) {
      PrimExpr source = tensorA[i][k] * tensorB[k][j];
      Var x("x", source.dtype()), y("y", source.dtype());
      PrimExpr result = tvm::tir::Add(x, y);
      PrimExpr identity_element = tvm::te::make_zero(source.dtype());  // NOLINT
      CommReducer combiner = CommReducer({x}, {y}, {result}, {identity_element});
      Reduce reduce{combiner, {source}, {k,},
                    tvm::te::make_const(DataType::Bool(4), true), 0, {}};

      LOG_PRINT_VAR(combiner);
      LOG_PRINT_VAR(combiner->lhs);
      LOG_PRINT_VAR(combiner->rhs);
      LOG_PRINT_VAR(combiner->result);
      LOG_PRINT_VAR(combiner->identity_element);
      LOG_PRINT_VAR(reduce->dtype);
      LOG_PRINT_VAR(reduce);
      LOG_PRINT_VAR(reduce->combiner);
      LOG_PRINT_VAR(reduce->source);
      LOG_PRINT_VAR(reduce->init)
      LOG_PRINT_VAR(reduce->axis);
      LOG_PRINT_VAR(reduce->condition);
      LOG_PRINT_VAR(reduce->value_index);
      LOG_PRINT_VAR(reduce->dtype);
      return reduce;
    };
    // clang-format on

    Array<PrimExpr> outShape{tensorA->GetShape()[0], tensorA->GetShape()[1]};
    /// @ref Please refer to tests/cpp/src/te/operation-test.cc for more details about
    /// `tvm::te::compute`.
    tvm::te::Tensor tensorD = tvm::te::compute(outShape, func, "matmul", "tagmatmul");

    LOG_PRINT_VAR(tensorD);
  }
}

}  // namespace expr_test

REGISTER_TEST_SUITE(expr_test::TirExprTest, tir_expr_test_TirExprTest);
REGISTER_TEST_SUITE(expr_test::TirBufferLoadTest, tir_expr_test_TirBufferLoadTest);
REGISTER_TEST_SUITE(expr_test::TirProducerLoadTest, tir_expr_test_TirProducerLoadTest);
REGISTER_TEST_SUITE(expr_test::TirRampTest, tir_expr_test_TirRampTest);
REGISTER_TEST_SUITE(expr_test::TirBroadcastTest, tir_expr_test_TirBroadcastTest);
REGISTER_TEST_SUITE(expr_test::TirLetTest, tir_expr_test_TirLetTest);
REGISTER_TEST_SUITE(expr_test::TirCallTest, tir_expr_test_TirCallTest);
REGISTER_TEST_SUITE(expr_test::TirShuffleTest, tir_expr_test_TirShuffleTest);
REGISTER_TEST_SUITE(expr_test::TirCommReducerTest, tir_expr_test_TirCommReducerTest);
REGISTER_TEST_SUITE(expr_test::TirReduceTest, tir_expr_test_TirReduceTest);
