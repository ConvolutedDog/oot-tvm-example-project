#include "ir/source-map-test.h"

#define LOG_PRINT_VAR(stmt) std::cout << #stmt << ": " << (stmt) << '\n';
#define LOG_SPLIT_LINE(stmt)                                                             \
  std::cout << "==============" << (stmt) << "==============\n";

namespace source_map_test {

std::ostream &operator<<(std::ostream &os, const std::vector<std::pair<int, int>> &vec) {
  for (auto &pair : vec)
    os << "(" << pair.first << ", " << pair.second << ") ";
  return os;
}

void SpanTest() {
  LOG_SPLIT_LINE("SpanTest");

  ObjectPtr<SourceNameNode> sourcenamenodeptr = make_object<SourceNameNode>();
  sourcenamenodeptr->name = "test.cc";
  SourceName sourcename(sourcenamenodeptr);
  LOG_PRINT_VAR(sourcename);

  Span span1(sourcename, 1, 2, 3, 4);
  LOG_PRINT_VAR(span1);

  Span span2(sourcename, 1, 2, 4, 5);
  LOG_PRINT_VAR(span2);

  Array<Span> arr{{span1, span2}};
  LOG_PRINT_VAR(arr);

  SequentialSpan seqspan1{arr};
  LOG_PRINT_VAR(seqspan1);

  SequentialSpan seqspan2{{span1, span2}};
  LOG_PRINT_VAR(seqspan2);
}

void SourceTest() {
  LOG_SPLIT_LINE("SourceTest");

  ObjectPtr<SourceNameNode> sourcenamenodeptr = make_object<SourceNameNode>();
  sourcenamenodeptr->name = "sourcenamenodeptr->name";
  SourceName sourcename(sourcenamenodeptr);

  Source source(sourcename, "source-source");
  LOG_PRINT_VAR(source);

  LOG_PRINT_VAR(source.get()->line_map);
  LOG_PRINT_VAR(source.get()->line_map.size());
  LOG_PRINT_VAR(source.GetLine(1));

  SourceMap sourcemap{{{sourcename, source}, }};
  LOG_PRINT_VAR(sourcemap);
}

}

void SpanTest() { source_map_test::SpanTest(); }
void SourceTest() { source_map_test::SourceTest(); }
