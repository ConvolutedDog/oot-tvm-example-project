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

std::ostream &operator<<(std::ostream &os, const BufferType &buffertype) {
  switch (buffertype) {
    case BufferType::kDefault: os << "kDefault"; break;
    case BufferType::kAutoBroadcast: os << "kAutoBroadcast"; break;
    default: os << "Unknown BufferType"; break;
  }
  return os;
}

void TirBufferTest() {
  LOG_SPLIT_LINE("TirBufferTest");

  /// Define a Buffer instance. A Bufer's data type is equal to the type of elements it
  /// stored.
  DataType dtype = DataType::Float(32, 4);  // float32x4
  /// The type of data must be `PointerType`.
  ///
  /// When binding the buffer to a variable, we need to specify the data type of the
  /// variable to be `tvm::PointerType` which points to `PrimType` elements.
  /// @todo (yangjianchao) Whether the pointered dtype of `Var` should be consistent with
  /// the type of the Buffer?
  Var data{
      "dataptr", PointerType{PrimType{dtype}, "global"}
  };
  /// This shape contains the shape as it is accessed by BufferLoad/BufferStore nodes, and
  /// used by the low-level code generators.
  Array<PrimExpr> shape{128, 128};  // 2D 128 x 128 matrix buffer
  Array<PrimExpr> strides{128, 1};  // Row-major. Here, users can also don't specify
                                    // strides. Just use `strides{}` here and call
                                    // `buffer.MakeStrideView()` will generate strides
                                    // automatically (default to row-major: [128, 1]).
                                    // NOLINTNEXTLINE
  PrimExpr elem_offset =
      PrimExpr(64);              // The offset in terms of number of dtype elements
                                 // (including lanes).
                                 // The starting offset of the Buffer. If elem_offset=8,
                                 // the buffer starts with data + 8 * sizeof(dtype).
  String buffer_name{"buffer"};  // NOLINT
  /// The alignment of data in bytes.
  int align = 64;  // Specify byte alignment requirements for Buffer data (e.g., 16 for
                   // 16-byte alignment).
  /// Factor of elem_offset field, elem_offset is guaranteed to be multiple of
  /// offset_factor. @ref https://www.zhihu.com/question/565420155
  // NOLINTNEXTLINE
  int offset_factor = 64;  // elem_offset is guaranteed to be multiple of offset_factor.

  /// Axis separators is used to split the input axes into multiple sub-axes, which will
  /// be reflected in the output axes. The axis separators should be chosen from 0~n-1,
  /// where n is the number of dimensions of the buffer. The order of the axis separators
  /// should be in increasing order.
  /// @todo Supplement more details about axis_separators.
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
  LOG_PRINT_VAR(buffer->data);                // "dataptr"
  LOG_PRINT_VAR(buffer->dtype);               // float32x4
  LOG_PRINT_VAR(buffer->shape);               // [128, 128]
  LOG_PRINT_VAR(buffer->axis_separators);     // []
  LOG_PRINT_VAR(buffer->strides);             // [128, 1]
  LOG_PRINT_VAR(buffer->elem_offset);         // 64
  LOG_PRINT_VAR(buffer->name);                // "buffer"
  LOG_PRINT_VAR(buffer->data_alignment);      // 64
  LOG_PRINT_VAR(buffer->offset_factor);       // 64
  LOG_PRINT_VAR(buffer->buffer_type);         // kDefault
  LOG_PRINT_VAR(buffer->DefaultIndexType());  // int32
  LOG_PRINT_VAR(
      buffer->ElemOffset({1, 2}));  // 1 * 128 + 2 + 64 = 194
                                    // row=1, column=2 (row-major), elem_offset = 64
  // `buffer.OffsetOf({1, 2})` actually call `buffer->ElemOffset({1, 2})`.
  LOG_PRINT_VAR(buffer.OffsetOf({1, 2}));  // [194]
  // Return the scope of the variable that the buffer was binded to.
  LOG_PRINT_VAR(buffer.scope());  // "global"

  /// @todo (yangjianchao) Supplement more detailed information about
  /// `GetFlattenedBuffer`.
  LOG_PRINT_VAR(buffer.GetFlattenedBuffer());

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
  /// Only when the buffer has no specified strides and the buffer's shape is not empty,
  /// `MakeStrideView` will generate strides automatically (default to row-major: [128,
  /// 1]).
  LOG_SPLIT_LINE("MakeStrideView");
  Buffer bufferstrided = buffer.MakeStrideView();
  const_cast<BufferNode *>(bufferstrided.get())->VisitAttrs(&serializer);
  /// Test a new buffer without specified strides.
  {
    LOG_SPLIT_LINE("New Buffer MakeStrideView");
    // No specified strides.
    Buffer buffer = Buffer(data, dtype, shape, {}, elem_offset, buffer_name, align,
                           offset_factor, BufferType::kDefault, axis_separators, Span{});
    LOG_PRINT_VAR(buffer->strides);  // []
    buffer = buffer.MakeStrideView();
    LOG_PRINT_VAR(buffer->strides);  // [128, 1]
  }

  /// Make a new symbolic buffer representing a slice of the buffer. The two axis slices
  /// both start at 64, and the extents are 32 and 64 respectively (shape of buffersliced
  /// becomes [18, 18]).
  LOG_SPLIT_LINE("MakeSlice");
  Buffer buffersliced = buffer.MakeSlice({64, 64}, {32, 64});
  const_cast<BufferNode *>(buffersliced.get())->VisitAttrs(&serializer);
  // The shape of slice becomes `extents`.
  LOG_PRINT_VAR(buffersliced->shape);  // [32, 64]
  // If you specified the strides for `buffer`, the slice will continued to use this
  // stride.
  LOG_PRINT_VAR(buffersliced->strides);      // [128, 1]
  LOG_PRINT_VAR(buffersliced->elem_offset);  // 64 * 128 + 64 + 64 = 8320
  // The `offset_factor` will become 0, because the `elem_offset` of the slice can be
  // random (determined by `begins`).
  LOG_PRINT_VAR(buffer->offset_factor);          // 0
  LOG_PRINT_VAR(buffersliced.OffsetOf({0, 0}));  // 8320

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

  /// @todo (yangjianchao) Supplement more details about access_ptr.
  LOG_SPLIT_LINE("access_ptr");
  /// @param access_mask : int
  ///   The access pattern MASK. Indicate whether the access will read or write to the
  ///   data content (1 for READ and 2 for WRITE).
  /// @param ptr_type : DataType, optional
  ///   The data type of the result pointer. Do not specify unless we want to cast pointer
  ///   to specific type.
  /// @param content_lanes: int, optional
  ///   The number of lanes for the data type. This value is greater than one for vector
  ///   types.
  /// @param offset: PrimExpr, optional
  ///   The offset of pointer. We can use it to offset by the number of elements from the
  ///   address of ptr.
  /// @param extent: PrmExpr, optional
  ///   The extent of pointer.
  int access_mask = 1;  // READ, NOLINT
  PrimExpr accessptr = buffer.access_ptr(access_mask);
  /// Output:
  ///   T.tvm_access_ptr(T.type_annotation("float32x4"), dataptr, 64, 16384, 1) // 2 for READ
  LOG_PRINT_VAR(accessptr);

  /// Create an Expr that does a vector load at begin index.
  /// @note Only the lanes of buffer > 1 can use `vload` or `vstore`.
  LOG_SPLIT_LINE("vload");
  PrimExpr vloadprimexpr = buffer.vload({2, 2}, dtype, Broadcast{tvm::Bool{true}, 4});
  LOG_PRINT_VAR(vloadprimexpr);

  /// Create a Stmt that does a vector store at begin index.
  LOG_SPLIT_LINE("vstore");
  /// @note Only the lanes of buffer > 1 can use `vload` or `vstore`.
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
