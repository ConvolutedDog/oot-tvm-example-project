#include "te/operation-test.h"
#include "test-func-registry.h"
#include "tvm/relax/expr.h"
#include <functional>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/var.h>

namespace operation_test {

void TePlaceholderOpTest() {
  LOG_SPLIT_LINE("TePlaceholderOpTest");

  /// `PlaceholderOp` has only one output tensor nad no input tensor.
  PlaceholderOp placeholderop{
      "placeholderop", {16, 3, 224, 224},
       DataType::Float(32, 4)
  };
  LOG_PRINT_VAR(placeholderop);
  /// Output:
  ///   placeholder(placeholderop, 0x1a7c110)

  LOG_PRINT_VAR(placeholderop->name);
  LOG_PRINT_VAR(placeholderop->tag);
  LOG_PRINT_VAR(placeholderop->attrs);
  LOG_PRINT_VAR(placeholderop->num_outputs());
  LOG_PRINT_VAR(placeholderop->output_dtype(0));
  LOG_PRINT_VAR(placeholderop->output_shape(0));
  LOG_PRINT_VAR(placeholderop->InputTensors());
  LOG_PRINT_VAR(placeholderop->shape);
  LOG_PRINT_VAR(placeholderop->dtype);

  /// placeholder
  LOG_SPLIT_LINE("placeholder");
  {
    /// Tensor placeholder(Array<PrimExpr> shape, DataType dtype, std::string name) {
    ///   return PlaceholderOp(name, shape, dtype).output(0);
    /// }
    Tensor tensor = placeholder({16, 3, 224, 224}, DataType::Float(32, 4), "placeholder");
    LOG_PRINT_VAR(tensor);
  }
}

void TeComputeOpTest() {
  LOG_SPLIT_LINE("TeComputeOpTest");

  /// `ComputeOpNode` has its own compute expressions: `Array<PrimExpr> body`, and its
  /// output size is the size of `body`. This means that each `PrimExpr` in the `body`
  /// of `ComputeOpNode` will generate an output tensor.
  /// `InputTensors` of `ComputeOpNode` is difficult to access:
  /// @code{.cpp}
  ///     Array<Tensor> ComputeOpNode::InputTensors() const {
  ///       Array<Tensor> ret;
  ///       std::unordered_set<Tensor> visited;
  ///       for (auto& e : body) {
  ///         tir::PostOrderVisit(e, [&ret, &visited](const ObjectRef& n) {
  ///           if (auto* pload = n.as<tir::ProducerLoadNode>()) {
  ///             Tensor t = Downcast<Tensor>(pload->producer);
  ///             if (!visited.count(t)) {
  ///               ret.push_back(t);
  ///               visited.insert(t);
  ///             }
  ///           }
  ///         });
  ///       }
  ///       return ret;
  ///     }
  /// @endcode
  /// The above function perform a post-order traversal (`PostOrderVisit`) of the
  /// expression to access all child nodes. If a child node can all be converted
  /// to `tir::ProducerLoadNode` means that it is reading input tensors, and then
  /// we can deduplicate all read tensors to calculate the total number of input
  /// tensors.

  /// @note The `PrimExpr`s in the `body` of `ComputeOpNode` should be sonsistent
  /// with being all `Reduction` or not.

  {
    /// @example Python frontend
    /// @code{.py}
    ///   n = tvm.runtime.convert(1024)
    ///   A = te.placeholder((n,), name="A")
    ///   B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    /// @endcode

    // clang-format off
    LOG_SPLIT_LINE("1-D ComputeOp");
    PrimExpr n = 1024;
    PlaceholderOp placeholder{"A", {n,}, DataType::Float(32)};
    tvm::tir::Var var{"ax0", placeholder->shape[0]->dtype};
    Array<tvm::tir::Var> args{{var}};
    Array<IterVar> axises = {
      IterVar{tvm::Range{tvm::IntImm{placeholder->shape[0]->dtype, 0},
                        placeholder->shape[0]}, var,
              tvm::tir::IterVarType::kDataPar}};
    std::function<PrimExpr(const Array<tvm::tir::Var>&)> fcompute = 
      [&placeholder](const Array<tvm::tir::Var>& indices) {
        tvm::tir::Var var = indices[0];
        return tvm::tir::ProducerLoad(placeholder.output(0), {var}) +
              tvm::tir::make_const(DataType::Float(32), 1.0);
      };
    ComputeOp computeop{"B", "tagB", {}, axises, {fcompute(args)}};
    LOG_PRINT_VAR(computeop);
    /// Output:
    ///   compute(B, body=[A[ax0] + T.float32(1.0)],
    ///           axis=[T.iter_var(ax0, T.Range(0, 1024), "DataPar", "")],
    ///           reduce_axis=[], tag=tagB, attrs={})
    // clang-format on
  }

  {
    // clang-format off
    LOG_SPLIT_LINE("2-D ComputeOp");
    PrimExpr m = 1024, n = 512;
    PlaceholderOp placeholder{"A", {m, n,}, DataType::Float(32)};
    tvm::tir::Var var0{"ax0", placeholder->shape[0]->dtype};
    tvm::tir::Var var1{"ax1", placeholder->shape[1]->dtype};
    Array<tvm::tir::Var> args{{var0, var1}};
    Array<IterVar> axises = {
      IterVar{tvm::Range{tvm::IntImm{placeholder->shape[0]->dtype, 0},
                         placeholder->shape[0]}, var0,
              tvm::tir::IterVarType::kDataPar},
      IterVar{tvm::Range{tvm::IntImm{placeholder->shape[1]->dtype, 0},
                         placeholder->shape[1]}, var1,
              tvm::tir::IterVarType::kDataPar},};
    std::function<PrimExpr(const Array<tvm::tir::Var>&)> fcompute = 
    [&placeholder](const Array<tvm::tir::Var>& indices) {
      tvm::tir::Var var0 = indices[0];
      tvm::tir::Var var1 = indices[1];
      return tvm::tir::ProducerLoad(placeholder.output(0), {var0, var1}) +
             tvm::tir::make_const(DataType::Float(32), 1.0);
    };
    ComputeOp computeop{"B", "tagB", {}, axises, {fcompute(args)}};
    LOG_PRINT_VAR(computeop);
    /// Output:
    ///   compute(B, body=[A[ax0, ax1] + T.float32(1.0)],
    ///           axis=[T.iter_var(ax0, T.Range(0, 1024), "DataPar", ""),
    ///                 T.iter_var(ax1, T.Range(0, 512), "DataPar", "")],
    ///           reduce_axis=[], tag=tagB, attrs={})
    // clang-format on
  }

  /// compute
  LOG_SPLIT_LINE("compute");
  {
    // clang-format off
    Tensor tensor = placeholder({1024,}, DataType::Float(32), "A");
    std::function<PrimExpr(const tvm::tir::Var &)> fcompute =
        [&tensor](const tvm::tir::Var &index) {
          return tvm::tir::ProducerLoad(tensor, {index,}) +
                 tvm::tir::make_const(DataType::Float(32), 1.0);
        };
    tensor = compute({1024,}, fcompute, "B", "tagB");
    LOG_PRINT_VAR(tensor);
    /// Output:
    ///   Tensor(shape=[1024], op.name=B)
    // clang-format on
  }

  /// compute
  LOG_SPLIT_LINE("compute");
  {
    // clang-format off
    Tensor tensor = placeholder({1024, 512}, DataType::Float(32), "A");
    std::function<PrimExpr(const Array<tvm::tir::Var> &)> fcompute =
        [&tensor](const Array<tvm::tir::Var> &indices) {
          tvm::tir::Var var0 = indices[0];
          tvm::tir::Var var1 = indices[1];
          return tvm::tir::ProducerLoad(tensor, {var0, var1}) +
                 tvm::tir::make_const(DataType::Float(32), 1.0);
        };
    tensor = compute(tensor->shape, fcompute, "B", "tagB");
    LOG_PRINT_VAR(tensor);
    /// Output:
    ///   Tensor(shape=[1024, 512], op.name=B)
    // clang-format on
  }

  /// compute
  LOG_SPLIT_LINE("compute");
  {
    // clang-format off
    Tensor tensor = placeholder({1024, 512}, DataType::Float(32), "A");
    std::function<Array<PrimExpr>(const Array<tvm::tir::Var> &)> fcompute =
        [&tensor](const Array<tvm::tir::Var> &indices) {
          tvm::tir::Var var0 = indices[0];
          tvm::tir::Var var1 = indices[1];
          return Array{tvm::tir::ProducerLoad(tensor, {var0, var1}) +
                       tvm::tir::make_const(DataType::Float(32), 1.0)};
        };
    Array<Tensor> tensorbatch = compute(tensor->shape, fcompute, "B", "tagB");
    LOG_PRINT_VAR(tensorbatch);
    /// Output:
    ///   [Tensor(shape=[1024, 512], op.name=B)]
    // clang-format on
  }
}

void TeScanOpTest() {
  LOG_SPLIT_LINE("TeScanOpTest");

  /// @ref python/tvm/te/operation.py

  /// @note init, update, state_placeholder must have same length. The number of input
  /// tensors of `ScanOp` is equal to the sum size of `init` and `update`. The number
  /// of output tensors of `ScanOp` is equal to the sum size of `update`.

  /// @example
  /// @code{.py}
  ///   m = te.var("m")
  ///   n = te.var("n")
  ///   X = te.placeholder((m, n), name="X")
  ///   s_state = te.placeholder((m, n))
  ///   s_init = te.compute((1, n), lambda _, i: X[0, i])
  ///   s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
  ///   res = tvm.te.scan(s_init, s_update, s_state, X)
  /// @endcode

  // clang-format off
  tvm::tir::Var m{"m", tvm::DataType::Int(32)};
  tvm::tir::Var n{"n", tvm::DataType::Int(32)};
  Tensor input = placeholder({m, n}, DataType::Float(32), "input");
  Tensor s_state = placeholder({m, n}, DataType::Float(32), "s_state");  // NOLINT
  std::function<PrimExpr(const Array<tvm::tir::Var> &)> fcompute_init =  // NOLINT
      [&](const Array<tvm::tir::Var> &indices) {
        return input(0, indices[1]);
      };
  Tensor s_init = compute({1, n}, fcompute_init, "s_init");  // NOLINT
  std::function<PrimExpr(const Array<tvm::tir::Var> &)> fcompute_update =  // NOLINT
      [&](const Array<tvm::tir::Var> &indices) {
        return s_state(indices[0] - 1, indices[1]) + input(indices[0], indices[1]);
      };
  Tensor s_update = compute({m, n}, fcompute_update, "s_update");  // NOLINT
  // clang-format on
  IterVar scan_axis =  // NOLINT
      IterVar(tvm::Range::FromMinExtent(Array<Tensor>{s_init}[0]->shape[0],
                                        Array<Tensor>{s_update}[0]->shape[0] -
                                            Array<Tensor>{s_init}[0]->shape[0]),
              tvm::tir::Var("scan.idx"), tvm::tir::IterVarType::kOrdered);
  ScanOp scanop{"scan",   "tagscan",  {},        scan_axis,
                {s_init}, {s_update}, {s_state}, {input}};
  LOG_PRINT_VAR(scanop);
  /// Output:
  ///   scan(scan, 0x189b030)

  /// scan
  LOG_SPLIT_LINE("scan");
  {
    // clang-format off
    tvm::tir::Var m{"m", tvm::DataType::Int(32)};
    tvm::tir::Var n{"n", tvm::DataType::Int(32)};
    Tensor input = placeholder({m, n}, DataType::Float(32), "input");
    Tensor s_state = placeholder({m, n}, DataType::Float(32), "s_state");  // NOLINT
    std::function<PrimExpr(const Array<tvm::tir::Var> &)> fcompute_init =  // NOLINT
        [&](const Array<tvm::tir::Var> &indices) { return input(0, indices[1]); };
    Tensor s_init = compute({1, n}, fcompute_init, "s_init");                // NOLINT
    std::function<PrimExpr(const Array<tvm::tir::Var> &)> fcompute_update =  // NOLINT
        [&](const Array<tvm::tir::Var> &indices) {
          return s_state(indices[0] - 1, indices[1]) + input(indices[0], indices[1]);
        };
    Tensor s_update = compute({m, n}, fcompute_update, "s_update");  // NOLINT
    Array<Tensor> scanop =
        scan({s_init}, {s_update}, {s_state}, {input}, "scanop", "tagscan", {});
    LOG_PRINT_VAR(scanop);
    /// Output:
    ///   [Tensor(shape=[m, n], op.name=scanop)]
    // clang-format on
  }
}

PrimExpr TestOp(const Array<tvm::tir::Var> &inputs) { return (int32_t)inputs.size(); }

TVM_REGISTER_OP("TestOp");

void TeExternOpTest() {
  LOG_SPLIT_LINE("TeExternOpTest");

  /// @example
  /// @code{.py}
  ///   nn = 1024
  ///   n = tvm.runtime.convert(nn)
  ///   A = te.placeholder((n,), name="A")
  ///   B = te.compute((n,), lambda i: A[i] + 1, name="B")
  ///   def extern_generator(ins, outs):
  ///       """Manually write the IR for the extern function, add pipeline."""
  ///       return tvm.tir.call_packed("my_extern_array_func2", ins[0], outs[0])
  ///   C = te.extern(B.shape, [B], extern_generator, name="C")
  /// @endcode

  // clang-format off
  PrimExpr n = 1024;
  Tensor tensor = placeholder({n,}, tvm::DataType::Float(32), "A");
  std::function<PrimExpr(const tvm::tir::Var &)> fcompute = 
      [&tensor](const tvm::tir::Var &index) {
        return tvm::tir::ProducerLoad(tensor, {index}) +
               tvm::tir::make_const(DataType::Float(32), 1.0);
      };
  tensor = compute({n,}, fcompute, "B");
  LOG_PRINT_VAR(tensor);
  /// Output:
  ///   Tensor(shape=[1024], op.name=B)
  Array<Buffer> input_placeholders = {  // NOLINT
    tvm::tir::decl_buffer({n,}, tvm::DataType::Float(32), "input_placeholders",
                          "global")
  };
  Array<Buffer> output_placeholders = {  // NOLINT
    tvm::tir::decl_buffer({n,}, tvm::DataType::Float(32), "output_placeholders",
                          "global")
  };

  tvm::Op testop = tvm::Op::Get("TestOp");
  std::function<PrimExpr(const Array<tvm::tir::Var>&)> fcomputeextern = 
      [&](const Array<tvm::tir::Var> &vars) {
        return tvm::tir::Call(DataType::Handle(), testop, {vars[0], vars[1]});
      };
  // `vm::tir::Var("inputX"), tvm::tir::Var("inputY")` may have no sense here.
  PrimExpr body = fcomputeextern({tvm::tir::Var("inputX"), tvm::tir::Var("inputY")});
  ExternOp externop{"C", "tagC", {}, {tensor},
                    input_placeholders, output_placeholders,
                    tvm::tir::Evaluate{body}};
  LOG_PRINT_VAR(externop);
  /// Output:
  ///   extern(C, 0x189e250)
  // clang-format on
}

void TeOtherFuncTest() {
  LOG_SPLIT_LINE("TeOtherFuncTest");

  tvm::tir::Var var{"var", DataType::Int(32)};
  LOG_PRINT_VAR(var);
  /// Output:
  ///   var

  tvm::tir::IterVar threadaxis = thread_axis({0, 512}, "thread_axis");
  LOG_PRINT_VAR(threadaxis);
  /// Output:
  ///   T.iter_var(thread_axis, T.Range(0, 512), "ThreadIndex", "thread_axis")

  tvm::tir::IterVar reduceaxis = reduce_axis({0, 512}, "reduce_axis");
  LOG_PRINT_VAR(reduceaxis);
  /// Output:
  ///   T.iter_var(reduce_axis, T.Range(0, 512), "CommReduce", "")
}

}  // namespace operation_test

REGISTER_TEST_SUITE(operation_test::TePlaceholderOpTest,
                    te_operation_test_TePlaceholderOpTest);
REGISTER_TEST_SUITE(operation_test::TeComputeOpTest, te_operation_test_TeComputeOpTest);
REGISTER_TEST_SUITE(operation_test::TeScanOpTest, te_operation_test_TeScanOpTest);
REGISTER_TEST_SUITE(operation_test::TeExternOpTest, te_operation_test_TeExternOpTest);
REGISTER_TEST_SUITE(operation_test::TeOtherFuncTest, te_operation_test_TeOtherFuncTest);
