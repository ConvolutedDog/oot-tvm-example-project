#include "tvm/ir/source_map.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/memory.h"
#include <tvm/runtime/object.h>

namespace source_map_test {

using tvm::SequentialSpan;
using tvm::SequentialSpanNode;
using tvm::Source;
using tvm::SourceMap;
using tvm::SourceMapNode;
using tvm::SourceName;
using tvm::SourceNameNode;
using tvm::SourceNode;
using tvm::Span;
using tvm::SpanNode;

using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::ObjectRef;

using tvm::runtime::make_object;

using tvm::runtime::Array;

}  // namespace source_map_test

void SpanTest();
void SourceTest();
