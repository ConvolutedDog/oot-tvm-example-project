#include "tir/stmt-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace stmt_test {

void TirBufferStoreTest() {
  LOG_SPLIT_LINE("TirBufferStoreTest");

  /// Define a Buffer. This is same to `buffer-test.cc`.
  DataType dtype = DataType::Float(32, 4);
  /// The type of data must be `PointerType`.
  Var data{
      "dataptr", PointerType{PrimType{dtype}, "global"}
  };
  Array<PrimExpr> shape{128, 128};     // 2D 128 x 128 matrix buffer
  Array<PrimExpr> strides{128, 1};     // Row-major. Here, users can also don't specify
                                       // strides. Just use `strides{}` here and call
                                       // `buffer.MakeStrideView()` will generate strides
                                       // automatically (default to row-major: [128, 1]).
                                       // NOLINTNEXTLINE
  PrimExpr elem_offset = PrimExpr(0);  // The starting offset of the Buffer.
                                       // If elem_offset=8, the buffer starts
                                       // with data + 8 * sizeof(dtype).
  String buffer_name{"buffer"};        // NOLINT
  int align = 64;  // Specify byte alignment requirements for Buffer data (e.g., 16 for
                   // 16-byte alignment).
  // NOLINTNEXTLINE
  int offset_factor = 64;  // elem_offset is guaranteed to be multiple of offset_factor.
  // NOLINTNEXTLINE
  Array<IntImm> axis_separators{};  // The separators between input axes when generating
                                    // flattened output axes.

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
  ///
  /// @todo (yangjianchao) Supplement more detailed information about `axis_separators`.
  Buffer buffer = Buffer(data, dtype, shape, strides, elem_offset, buffer_name, align,
                         offset_factor, BufferType::kDefault, axis_separators, Span{});

  Var input = Var("input", DataType::Float(32));
  PrimExpr value = PrimExpr{input}, lanes = 4;
  Broadcast broadcast{value, lanes};
  Var var1 = Var("arg1", DataType::Int(32));
  Var var2 = Var("arg2", DataType::Int(32));
  Array<PrimExpr> indices = {var1, var2};  // buffer[var1][var2] = broadcast;

  LOG_SPLIT_LINE("BufferStore");
  BufferStore bufferstore{buffer, broadcast, indices};
  LOG_PRINT_VAR(bufferstore);
  /// Output:
  ///   buffer = T.Buffer((128, 128), "float32x4", strides=(128, 1), offset_factor=64)
  ///   input = T.float32()
  ///   arg1 = T.int32()
  ///   arg2 = T.int32()
  ///   buffer[arg1, arg2] = T.Broadcast(input, 4)

  /// Annotate the region where the buffer need to be read and write in the body. We only
  /// need to allocate the space for the corresponding region.
  LOG_SPLIT_LINE("BufferRealize");
  BufferRealize bufferrealize{
      buffer,
      {
        {var1 * 128, var1 * 128 + var2},
        },
      tvm::tir::const_true(1),
      bufferstore
  };
  LOG_PRINT_VAR(bufferrealize);
  /// Output:
  ///   buffer = T.Buffer((128, 128), "float32x4", strides=(128, 1), offset_factor=64)
  ///   arg1 = T.int32()
  ///   arg2 = T.int32()
  ///   with T.realize(buffer[arg1:arg1 + (arg2 - arg1)]):
  ///       input = T.float32()
  ///       buffer[arg1, arg2] = T.Broadcast(input, 4)

  /// DeclBuffer: Declare a buffer that can be used in the body.
  LOG_SPLIT_LINE("DeclBuffer");
  Evaluate evaluate{buffer->elem_offset};
  DeclBuffer declbuffer{buffer, evaluate};
  LOG_PRINT_VAR(declbuffer);
  /// Output:
  ///   dataptr = T.handle("float32x4", "global")
  ///   with T.decl_buffer((128, 128), "float32x4", data=dataptr,
  ///                      strides=(128, 1), offset_factor=64) as buffer:
  ///       T.evaluate(0)

  /// SeqStmt
  LOG_SPLIT_LINE("SeqStmt");
  SeqStmt seqstmt{
      {bufferstore, bufferrealize, declbuffer, evaluate}
  };
  LOG_PRINT_VAR(seqstmt);
  /// Output:
  ///   buffer = T.Buffer((128, 128), "float32x4", strides=(128, 1), offset_factor=64)
  ///   input = T.float32()
  ///   arg1 = T.int32()
  ///   arg2 = T.int32()
  ///   buffer[arg1, arg2] = T.Broadcast(input, 4)
  ///   with T.realize(buffer[arg1 * 128:arg1 * 128 + (arg1 * 128 + arg2 - arg1 * 128)]):
  ///       buffer[arg1, arg2] = T.Broadcast(input, 4)
  ///   with T.decl_buffer((128, 128), "float32x4", data=buffer.data, strides=(128, 1),
  ///                      offset_factor=64) as buffer:
  ///       T.evaluate(0)
  ///   T.evaluate(0)
}

void TirProducerStoreTest() {
  LOG_SPLIT_LINE("TirProducerStoreTest");

  Array<PrimExpr> shape{128, 128};
  DataType dtype = DataType::BFloat(16, 4);
  Tensor tensor{
      shape, dtype, PlaceholderOp{"placeholder", shape, dtype},
        0
  };
  LOG_PRINT_VAR(tensor);
  /// Output: Tensor(shape=[128, 128], op.name=placeholder)

  Var input = Var("input", DataType::Float(32));
  PrimExpr value = PrimExpr{input}, lanes = 4;
  Broadcast broadcast{value, lanes};
  Var var1 = Var("arg1", DataType::Int(32));
  Var var2 = Var("arg2", DataType::Int(32));
  Array<PrimExpr> indices = {var1, var2};  // buffer[var1][var2] = broadcast;

  LOG_SPLIT_LINE("ProducerStore");
  ProducerStore producerstore{tensor, broadcast, indices};
  LOG_PRINT_VAR(producerstore);
  /// Output:
  ///   arg1 = T.int32()
  ///   arg2 = T.int32()
  ///   input = T.float32()
  ///   placeholder[arg1, arg2] = T.Broadcast(input, 4)

  LOG_SPLIT_LINE("ProducerRealize");
  ProducerRealize producerrealize{
      tensor,
      {
        {var1, var2},
        },
      tvm::tir::const_true(1),
      producerstore
  };
  LOG_PRINT_VAR(producerrealize);
  /// Output;
  ///   arg1 = T.int32()
  ///   arg2 = T.int32()
  ///   with T.ProducerRealize(placeholder[arg1:arg1 + (arg2 - arg1)], T.bool(True)):
  ///     input = T.float32()
  ///     placeholder[arg1, arg2] = T.Broadcast(input, 4)

  /// Let
  LOG_SPLIT_LINE("Let");
  Var varlet = Var("letvar");
  Evaluate evaluatevarlet{varlet};
  LetStmt letstmt{varlet, 1, evaluatevarlet};
  LOG_PRINT_VAR(letstmt);
  /// Output:
  ///   with T.LetStmt(1) as letvar:
  ///     T.evaluate(letvar)

  /// AttrStmt
  LOG_SPLIT_LINE("AttrStmt");
  AttrStmt attrstmt{tensor, "attrstmt", tvm::tir::const_true(1), evaluatevarlet};
  LOG_PRINT_VAR(attrstmt);
  /// Output:
  ///   with T.attr(metadata["Tensor"][0], "attrstmt", T.bool(True)):
  ///     letvar = T.int32()
  ///     T.evaluate(letvar)

  /// AssertStmt
  LOG_SPLIT_LINE("AssertStmt");
  Var var1assert{"var1"}, var2assert{"var2"};
  Evaluate evaluateassert{var1assert};
  AssertStmt assertsmtmt{
      tvm::tir::LT{var1assert, var2assert},
      tvm::tir::StringImm{"assert message"},
      evaluateassert
  };
  LOG_PRINT_VAR(assertsmtmt);
  /// Output:
  ///   var1 = T.int32()
  ///   var2 = T.int32()
  ///   with T.Assert(var1 < var2, "assert message"):
  ///       T.evaluate(var1)
}

void TirAllocateTest() {
  LOG_SPLIT_LINE("TirAllocateTest");

  DataType dtype = DataType::Float(32, 4);
  Var buffervar{"buffer", PointerType{PrimType{dtype}}};
  Array<PrimExpr> extents{128, 128};
  PrimExpr condition = tvm::tir::const_true(1);

  Evaluate evaluate{buffervar};

  /// Allocate
  LOG_SPLIT_LINE("Allocate");
  Allocate allocate{buffervar, dtype, extents, condition, evaluate};
  LOG_PRINT_VAR(allocate);

  /// AllocateConst
  LOG_SPLIT_LINE("AllocateConst");
  tvm::runtime::NDArray array =
      tvm::runtime::NDArray::Empty({3, 3}, dtype, tvm::Device{DLDeviceType::kDLCPU, 0});
  AllocateConst allocateconst{buffervar, dtype, extents, array, evaluate};
  LOG_PRINT_VAR(allocateconst);
  /// Output:
  /// with T.allocate_const([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  ///                       "float32x4", [128, 128]) as buffer:
  ///   T.evaluate(buffer)
}

void TirForTest() {
  LOG_SPLIT_LINE("TirForTest");

  // int start, end;
  // int middle;
  // for (int i = start; i < end; ++i) {
  //   if (i < middle)
  //     evaluate(i);
  //   else
  //     evaluate(i * 2);
  // }

  /// For
  LOG_SPLIT_LINE("For");
  Var start{"start", DataType::Int(32)};
  Var end{"end", DataType::Int(32)};
  Var middle{"middle", DataType::Int(32)};
  Var loopvar{"i", DataType::Int(32)};
  PrimExpr extent = end - start;
  ForKind forkind{ForKind::kSerial};
  IfThenElse ifelsestmt{
      tvm::tir::LT{loopvar, middle},
      Evaluate{loopvar},
      Evaluate{loopvar * 2}
  };
  For forstmt{loopvar, start, extent, forkind, ifelsestmt};
  LOG_PRINT_VAR(forstmt);
  /// Output:
  ///   for i in range(start, start + (end - start)):
  ///     start = T.int32()
  ///     end = T.int32()
  ///     middle = T.int32()
  ///     if i < middle:
  ///         T.evaluate(i)
  ///     else:
  ///         T.evaluate(i * 2)

  /// While
  LOG_SPLIT_LINE("While");
  loopvar = Var{"i", DataType::Int(32)};
  ifelsestmt = IfThenElse{
      tvm::tir::LT{loopvar, middle},
      Evaluate{loopvar},
      Evaluate{loopvar * 2}
  };
  While whilestmt{
      tvm::tir::And{tvm::tir::LT{loopvar, end}, tvm::tir::GE{loopvar, start}},
      ifelsestmt
  };
  LOG_PRINT_VAR(whilestmt);
  /// Output:
  ///   i = T.int32()
  ///   end = T.int32()
  ///   start = T.int32()
  ///   while i < end and i >= start:
  ///       middle = T.int32()
  ///       if i < middle:
  ///           T.evaluate(i)
  ///       else:
  ///           T.evaluate(i * 2)
}

void TirPrefetchTest() {
  LOG_SPLIT_LINE("TirPrefetchTest");

  /// Define a Buffer. This is same to `buffer-test.cc`.
  DataType dtype = DataType::Float(32, 4);
  /// The type of data must be `PointerType`.
  Var data{
      "dataptr", PointerType{PrimType{dtype}, "global"}
  };
  Array<PrimExpr> shape{128, 128};     // 2D 128 x 128 matrix buffer
  Array<PrimExpr> strides{128, 1};     // Row-major. Here, users can also don't specify
                                       // strides. Just use `strides{}` here and call
                                       // `buffer.MakeStrideView()` will generate strides
                                       // automatically (default to row-major: [128, 1]).
                                       // NOLINTNEXTLINE
  PrimExpr elem_offset = PrimExpr(0);  // The starting offset of the Buffer.
                                       // If elem_offset=8, the buffer starts
                                       // with data + 8 * sizeof(dtype).
  String buffer_name{"buffer"};        // NOLINT
  int align = 64;  // Specify byte alignment requirements for Buffer data (e.g., 16 for
                   // 16-byte alignment).
  // NOLINTNEXTLINE
  int offset_factor = 64;  // elem_offset is guaranteed to be multiple of offset_factor.
  // NOLINTNEXTLINE
  Array<IntImm> axis_separators{};  // The separators between input axes when generating
                                    // flattened output axes.

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
  ///
  /// @todo (yangjianchao) Supplement more detailed information about `axis_separators`.
  Buffer buffer = Buffer(data, dtype, shape, strides, elem_offset, buffer_name, align,
                         offset_factor, BufferType::kDefault, axis_separators, Span{});

  // Buffer[0:1][28:29].
  Array<tvm::Range> bounds{
      {
       {0, 1},
       {28, 29},
       }
  };

  /// Prefetch
  LOG_SPLIT_LINE("Prefetch");
  Prefetch prefetch = Prefetch(buffer, bounds);
  LOG_PRINT_VAR(prefetch);
  /// Output:
  ///   buffer = T.Buffer((128, 128), "float32x4", strides=(128, 1), offset_factor=64)
  ///   T.prefetch(buffer, [T.Range(0, 1), T.Range(28, 29)])

  /// BufferRegion
  LOG_SPLIT_LINE("BufferRegion");
  LOG_PRINT_VAR(BufferRegion::FullRegion(buffer));
  /// Output:
  ///   FullRegion(buffer): buffer[0:128, 0:128]

  // Buffer[1:23][2:4].
  Array<tvm::Range> region{
      {{1, 23}, {2, 4}}
  };
  BufferRegion bufferregion = BufferRegion(buffer, region);
  LOG_PRINT_VAR(bufferregion);
  /// Output:
  ///   buffer[1:23, 2:4]

  /// MatchBufferRegion
  LOG_SPLIT_LINE("MatchBufferRegion");
  Array<tvm::Range> matchregion{
      {{0, 128}, {0, 128}}
  };
  BufferRegion newbufferregion = BufferRegion(buffer, matchregion);
  MatchBufferRegion matchbufferregion = MatchBufferRegion(buffer, newbufferregion);
  LOG_PRINT_VAR(matchbufferregion);
  /// Output:
  ///   buffer = T.match_buffer(buffer[0:128, 0:128], (128, 128), "float32x4",
  ///                           strides=(128, 1), offset_factor=64)

  /// Block
  LOG_SPLIT_LINE("Block");
  IterVar i = IterVar(tvm::Range(0, 128), Var{"i"}, tvm::tir::IterVarType::kDataPar);
  IterVar j = IterVar(tvm::Range(0, 128), Var{"j"}, tvm::tir::IterVarType::kDataPar);
  // clang-format off
  Block block = Block({i,j}, {BufferRegion(buffer, {{0, i}, {0, j}}), },
                      {BufferRegion(buffer, {{64, PrimExpr{128}+i}, {16, PrimExpr{64}+j}}), },
                      "block", Evaluate{1});
  // clang-format on
  LOG_PRINT_VAR(block);
  /// Output:
  ///   with T.block("block", no_realize=True):
  ///     i = T.axis.spatial(128)
  ///     j = T.axis.spatial(128)
  ///     buffer = T.Buffer((128, 128), "float32x4", strides=(128, 1), offset_factor=64)
  ///     T.reads(buffer[0:i, 0:j])
  ///     T.writes(buffer[64:64 + (128 + i - 64), 16:16 + (64 + j - 16)])
  ///     T.evaluate(1)

  /// BlockRealize
  /// @todo (yangjianchao) Supplement BlockRealize.
  LOG_SPLIT_LINE("BlockRealize");
  BlockRealize blockrealize{
      {i, j},
      tvm::Bool{1},
      block
  };
  LOG_PRINT_VAR(blockrealize);
}

void TirTypeAnnotationTest() {
  LOG_SPLIT_LINE("TirTypeAnnotationTest");

  LOG_PRINT_VAR(TypeAnnotation(DataType::Float(32)));
  /// Output: T.type_annotation("float32")
}

}  // namespace stmt_test

REGISTER_TEST_SUITE(stmt_test::TirBufferStoreTest, tir_stmt_test_TirBufferStoreTest);
REGISTER_TEST_SUITE(stmt_test::TirProducerStoreTest, tir_stmt_test_TirProducerStoreTest);
REGISTER_TEST_SUITE(stmt_test::TirAllocateTest, tir_stmt_test_TirAllocateTest);
REGISTER_TEST_SUITE(stmt_test::TirForTest, tir_stmt_test_TirForTest);
REGISTER_TEST_SUITE(stmt_test::TirPrefetchTest, tir_stmt_test_TirPrefetchTest);
REGISTER_TEST_SUITE(stmt_test::TirTypeAnnotationTest,
                    tir_stmt_test_TirTypeAnnotationTest);
