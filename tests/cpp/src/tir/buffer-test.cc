#include "tir/buffer-test.h"
#include "test-func-registry.h"
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

namespace buffer_test {

class MyIRSerializer : public AttrVisitor {
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

void TirBufferTest() {
  LOG_SPLIT_LINE("TirBufferTest");

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

  MyIRSerializer serializer;
  LOG_SPLIT_LINE("Buffer");
  const_cast<BufferNode *>(buffer.get())->VisitAttrs(&serializer);

  /// Construct a new buffer given shape, and dtype.
  LOG_SPLIT_LINE("decl_buffer");
  Buffer bufferdecl =
      decl_buffer(shape, dtype, buffer_name, "global", axis_separators, Span{});
  const_cast<BufferNode *>(bufferdecl.get())->VisitAttrs(&serializer);

  /// Creates TIR Buffer for provided parameters.
  LOG_SPLIT_LINE("BufferWithOffsetAlignment");
  bool compact = true;  /// If the statement has already bound to a compact buffer.
                        /// @todo (yangjianchao) Supplement more details about `compact`.
  Buffer bufferwithoffsetalignment = BufferWithOffsetAlignment(
      shape, dtype, buffer_name, align, offset_factor, compact, "global");
  const_cast<BufferNode *>(bufferwithoffsetalignment.get())->VisitAttrs(&serializer);

  /// Return a new buffer that is equivalent with current one but always add stride field.
  LOG_SPLIT_LINE("MakeStrideView");
  Buffer bufferstrided = buffer.MakeStrideView();
  const_cast<BufferNode *>(bufferstrided.get())->VisitAttrs(&serializer);

  /// Make a new symbolic buffer representing a slice of the buffer. The two axis slices
  /// both start at 64, and the extents are 32 and 64 respectively (shape of buffersliced
  /// becomes [18, 18]).
  LOG_SPLIT_LINE("MakeSlice");
  Buffer buffersliced = buffer.MakeSlice({64, 64}, {32, 64});
  const_cast<BufferNode *>(buffersliced.get())->VisitAttrs(&serializer);

  /// Get a flattened version of the buffer.
  LOG_SPLIT_LINE("GetFlattenedBuffer");
  Buffer bufferflattend = buffer.GetFlattenedBuffer();
  const_cast<BufferNode *>(bufferflattend.get())->VisitAttrs(&serializer);

  /// Determine the offset in the buffer of the given index.
  LOG_SPLIT_LINE("OffsetOf");
  LOG_PRINT_VAR(buffer.OffsetOf({0, 1}));  // [1]
  LOG_PRINT_VAR(buffer.OffsetOf({1, 2}));  // [130] = [1 * 128 + 2]

  /// Return the storage scope associated with this buffer.
  LOG_SPLIT_LINE("scope()");
  LOG_PRINT_VAR(buffer.scope());

  /// @todo (yangjianchao) Supplement test for access_ptr.
  LOG_SPLIT_LINE("access_ptr");
  int access_mask = 0x0001;  // NOLINT
  PrimExpr accessptr = buffer.access_ptr(access_mask);
  LOG_PRINT_VAR(accessptr);

  /// Create an Expr that does a vector load at begin index.
  LOG_SPLIT_LINE("vload");
  PrimExpr vloadprimexpr = buffer.vload({2, 2}, dtype, Broadcast{tvm::Bool{true}, 4});
  LOG_PRINT_VAR(vloadprimexpr);

  /// Create a Stmt that does a vector store at begin index.
  LOG_SPLIT_LINE("vstore");
  Stmt stmt = buffer.vstore({3, 3}, Broadcast{1.0f, 4}, Broadcast{tvm::Bool{true}, 4});
  LOG_PRINT_VAR(stmt);
}

void TirDataProducerTest() {
  LOG_SPLIT_LINE("DataProducer");
  LOG_PRINT_VAR("Refer to the test of `tvm::te::Tensor`");
}

}  // namespace buffer_test

REGISTER_TEST_SUITE(buffer_test::TirBufferTest, tir_buffer_test_TirBufferTest);
REGISTER_TEST_SUITE(buffer_test::TirDataProducerTest,
                    tir_buffer_test_TirDataProducerTest);
