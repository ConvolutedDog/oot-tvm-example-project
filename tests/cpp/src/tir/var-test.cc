#include "tir/var-test.h"
#include "dlpack/dlpack.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>

namespace var_test {

void TirVarTest() {
  LOG_SPLIT_LINE("VarTest");
  /// `Var` instances can be constructed from two base constructors:
  /// 1. TVM_DLL explicit Var(String name_hint = "v", DataType dtype = DataType::Int(32),
  ///                         Span span = Span());
  /// 2. TVM_DLL explicit Var(String name_hint, Type type_annotation, Span span = Span());
  ///
  /// The first constructor provides a simple type annotation, and in the internal imple-
  /// mentation, `DataType` is converted to `Type` by `GetTypeFromRuntimeDataType()`.
  /// The second constructor provides a more detailed type annotation, and in the internal
  /// implementation, `Type` is converted to `DataType` by `GetRuntimeDataType()`.
  ///
  /// The corresponding relation:
  /// 1. DataTypeInst.code() == DataType::KHandle &&
  ///    DataTypeInst.bits() == 0 && DataTypeInst.lanes() == 0
  ///    => TupleType(Array<Type>())
  /// 2. Others DataTypeInsts => PrimType(DataTypeInst)
  /// 3. tvm::PointerType => DataType(kHandle, bits, lanes)
  Var x{"var", DataType::UInt(32, 1, false)};
  LOG_PRINT_VAR(x->name_hint);        // "var"
  LOG_PRINT_VAR(x->dtype);            // uint32
  LOG_PRINT_VAR(x->type_annotation);  // T.uint32
  LOG_BLANK_LINE;

  Var xx{"var", tvm::PrimType{DataType::UInt(32, 1)}};
  LOG_PRINT_VAR(xx->name_hint);        // "var"
  LOG_PRINT_VAR(xx->dtype);            // uint32
  LOG_PRINT_VAR(xx->type_annotation);  // T.uint32

  Var y = x.copy_with_name("varcopy_with_name");
  LOG_PRINT_VAR(y->name_hint);        // => "varcopy_with_name"
  LOG_PRINT_VAR(y->dtype);            // default to uint32
  LOG_PRINT_VAR(y->type_annotation);  // default to T.uint32
  LOG_BLANK_LINE;

  Var z = x.copy_with_suffix("suffix");
  LOG_PRINT_VAR(z->name_hint);        // => "varsuffix"
  LOG_PRINT_VAR(z->dtype);            // default to uint32
  LOG_PRINT_VAR(z->type_annotation);  // default to T.uint32
  LOG_BLANK_LINE;

  Var w = x.copy_with_dtype(DataType::Int(16, 1));
  LOG_PRINT_VAR(w->name_hint);        // default to "var"
  LOG_PRINT_VAR(w->dtype);            // => int16
  LOG_PRINT_VAR(w->type_annotation);  // => T.int16
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.get() == xx.get());  // False
  LOG_PRINT_VAR(x.get() == y.get());   // False
}

void TirSizeVarTest() {
  LOG_SPLIT_LINE("TirSizeVarTest");
  /// Similar to `Var`, `SizeVar` can be constructed from two base constructors:
  /// 1. TVM_DLL explicit SizeVar(String name_hint = "s", DataType t = DataType::Int(32),
  ///                             Span span = Span());
  /// 2. TVM_DLL explicit SizeVar(String name_hint, Type type_annotation,
  ///                             Span span = Span());
  SizeVar x{"sizevar", DataType::UInt(32, 1, false)};
  LOG_PRINT_VAR(x->name_hint);
  LOG_PRINT_VAR(x->dtype);
  LOG_PRINT_VAR(x->type_annotation);
  LOG_BLANK_LINE;

  Var y = x.copy_with_name("varcopy_with_name");
  LOG_PRINT_VAR(y->name_hint);
  LOG_PRINT_VAR(y->dtype);
  LOG_PRINT_VAR(y->type_annotation);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.get() == y.get());
  LOG_BLANK_LINE;

  Var z = x.copy_with_suffix("suffix");
  LOG_PRINT_VAR(z->name_hint);
  LOG_PRINT_VAR(z->dtype);
  LOG_PRINT_VAR(z->type_annotation);
  LOG_BLANK_LINE;

  Var w = x.copy_with_dtype(DataType::Int(32, 1));
  LOG_PRINT_VAR(w->name_hint);
  LOG_PRINT_VAR(w->dtype);
  LOG_PRINT_VAR(w->type_annotation);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.get() == w.get());
  LOG_BLANK_LINE;

  SizeVar o{"sizevar", PointerType{VoidType()}};
  LOG_PRINT_VAR(o->name_hint);
  LOG_PRINT_VAR(o->dtype);            // => handle
  LOG_PRINT_VAR(o->type_annotation);  // => T.handle(None)
  LOG_BLANK_LINE;
}

/// An iteration variable representing an iteration over a one dimensional interval.
/// The dtype of the extent of the `dom` of the `IterVar` must match the dtype of the
/// internal `Var`.
void TirIterVarTest() {
  LOG_SPLIT_LINE("TirIterVarTest");

  PrimExpr x = 0;
  PrimExpr y = 4;

  LOG_PRINT_VAR(x.as<IntImmNode>()->value);
  LOG_PRINT_VAR(y.as<IntImmNode>()->value);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.as<PrimExprNode>()->dtype);
  LOG_PRINT_VAR(y.as<PrimExprNode>()->dtype);
  LOG_BLANK_LINE;

  LOG_PRINT_VAR(x.as<BaseExprNode>()->GetTypeKey());
  LOG_PRINT_VAR(y.as<BaseExprNode>()->GetTypeKey());
  LOG_BLANK_LINE;

  Range range{x, y};
  /// Here, the dtype of range is Int (because PrimExpr x = 0 and y = 4,
  /// 4 is an integer). So the dtype of Var should also be defined as Int,
  /// otherwise, the initialization of IterVar will fail. Another point is
  /// that the dtype should always be Int. @ref
  /// https://github.com/apache/tvm/blob/4ef582a3319f30fac2716091f835e493ec161ffd/src/tir/ir/expr.cc#L170
  /// https://github.com/apache/tvm/blob/4ef582a3319f30fac2716091f835e493ec161ffd/src/tir/ir/expr.cc#L174
  DataType dtype = DataType::Int(32, 1);
  Var var{"var", dtype};

  /// @brief A common usage of `IterVar` is being axes, such as `Reduce` and `ComputeOp`.
  ///
  /// @brief IterVarType:
  ///   /*!
  ///    * \brief Type of iteration variable.
  ///    *  Each IterVar have a specific type.
  ///    *
  ///    *  The type of iter var can be overriden via
  ///    *  stage.iter_var_attrs given they are compatible.
  ///    */
  ///   enum IterVarType : int {
  ///     /*!
  ///      * \brief Data parallel iteration.
  ///      *  This normally corresponds to axis of Tensor.
  ///      *  Allow all IterVar manipulations.
  ///      *
  ///      * \note This does not mean the loop
  ///      *  have to be executed in parallel fashion.
  ///      */
  ///     kDataPar = 0,
  ///     /*!
  ///      * \brief The IterVar itself is a thread-index
  ///      *  of a fixed thread launching group.
  ///      *  Note that this is already assumed to be parallelized.
  ///      *
  ///      *  Disallow: split/fuse/vectorize/parallel
  ///      */
  ///     kThreadIndex = 1,
  ///     /*!
  ///      * \brief Communicative reduction.
  ///      *  Cannot be directly parallelized.
  ///      *
  ///      *  Disallow: parallel/vectorize
  ///      */
  ///     kCommReduce = 2,
  ///     /*!
  ///      * \brief Serial loops with loop carry dependency,
  ///      *  the iteration must execute in order.
  ///      *  Cannot be re-ordered.
  ///      *
  ///      *  Disallow: reorder/parallel/vectorize
  ///      */
  ///     kOrdered = 3,
  ///     /*!
  ///      * \brief IterVar is opaque,
  ///      *
  ///      *  May not corresponds to any generated loop
  ///      *  Disallow all IterVar manipulations and compute_at
  ///      *
  ///      * \note This is usually used to implement composite op
  ///      *  or external op, where the
  ///      */
  ///     kOpaque = 4,
  ///     // The following are possible additional
  ///     // types that are provided during schedule
  ///     /*!
  ///      * \brief The execution is unrolled.
  ///      */
  ///     kUnrolled = 5,
  ///     /*!
  ///      * \brief The loop is vectorized.
  ///      */
  ///     kVectorized = 6,
  ///     /*!
  ///      * \brief The loop is parallelized.
  ///      */
  ///     kParallelized = 7,
  ///     /*!
  ///      * \brief Marks boundary of tensorization intrinsic.
  ///      */
  ///     kTensorized = 8
  ///   };
  ///
  /// @brief Supplement more details about usage of `IterVar`.
  IterVar itervar{range, var, IterVarType::kOrdered, String("thread_tag")};

  LOG_PRINT_VAR(itervar->dom);
  LOG_PRINT_VAR(itervar->dom->extent.defined());
  LOG_PRINT_VAR(itervar->var);
  LOG_PRINT_VAR(itervar->iter_type);
  LOG_PRINT_VAR(IterVarType2String(itervar->iter_type));
  // Additional tag on the iteration variable, set this if this is bound already
  // to a known thread tag.
  LOG_PRINT_VAR(itervar->thread_tag);
  LOG_PRINT_VAR(itervar->span);

  LOG_PRINT_VAR(itervar.operator PrimExpr());

  LOG_PRINT_VAR(itervar.as<IterVarNode>()->dom);
}

}  // namespace var_test

REGISTER_TEST_SUITE(var_test::TirVarTest, tir_var_test_TirVarTest);
REGISTER_TEST_SUITE(var_test::TirSizeVarTest, tir_var_test_TirSizeVarTest);
REGISTER_TEST_SUITE(var_test::TirIterVarTest, tir_var_test_TirIterVarTest);
