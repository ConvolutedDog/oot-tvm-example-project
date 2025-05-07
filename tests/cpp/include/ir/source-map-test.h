#include "tvm/ir/source_map.h"
#include "tvm/runtime/memory.h"
#include <tvm/runtime/object.h>
#include "tvm/runtime/container/array.h"

namespace source_map_test {
  
using tvm::SourceNameNode;
using tvm::SourceName;
using tvm::SpanNode;
using tvm::Span;
using tvm::SequentialSpanNode;
using tvm::SequentialSpan;
using tvm::SourceNode;
using tvm::Source;
using tvm::SourceMapNode;
using tvm::SourceMap;

using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::ObjectRef;

using tvm::runtime::make_object;

using tvm::runtime::Array;

void SpanTest();
void SourceTest();

}

void SpanTest();
void SourceTest();
