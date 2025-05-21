#include "tvm/tir/expr_functor.h"
#include "tvm/tir/op.h"
#include <tvm/runtime/data_type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/var.h>

namespace expr_functor_test {

using tvm::tir::ExprFunctor;
using tvm::tir::ExprMutator;
using tvm::tir::ExprVisitor;

using tvm::DataType;
using tvm::floor;
using tvm::PrimExpr;
using tvm::Range;
using tvm::Span;
using tvm::runtime::Array;
using tvm::tir::Buffer;
using tvm::tir::BufferLoad;
using tvm::tir::const_true;
using tvm::tir::decl_buffer;
using tvm::tir::IterVar;
using tvm::tir::IterVarType;
using tvm::tir::make_const;
using tvm::tir::MakeConstScalar;
using tvm::tir::Var;

using tvm::runtime::Object;
using tvm::tir::AddNode;
using tvm::tir::AndNode;
using tvm::tir::BroadcastNode;
using tvm::tir::BufferLoadNode;
using tvm::tir::CallNode;
using tvm::tir::CastNode;
using tvm::tir::DivNode;
using tvm::tir::EQNode;
using tvm::tir::FloatImmNode;
using tvm::tir::FloorDivNode;
using tvm::tir::FloorModNode;
using tvm::tir::GENode;
using tvm::tir::GTNode;
using tvm::tir::IntImmNode;
using tvm::tir::LENode;
using tvm::tir::LetNode;
using tvm::tir::LTNode;
using tvm::tir::MaxNode;
using tvm::tir::MinNode;
using tvm::tir::ModNode;
using tvm::tir::MulNode;
using tvm::tir::NENode;
using tvm::tir::NotNode;
using tvm::tir::OrNode;
using tvm::tir::ProducerLoadNode;
using tvm::tir::RampNode;
using tvm::tir::ReduceNode;
using tvm::tir::SelectNode;
using tvm::tir::ShuffleNode;
using tvm::tir::SizeVarNode;
using tvm::tir::StringImmNode;
using tvm::tir::SubNode;
using tvm::tir::VarNode;

using tvm::runtime::DLDataTypeCode2Str;
using tvm::runtime::String;

class StdCoutExprVisitor : public ExprFunctor<String(const PrimExpr &)> {
public:
  using R = ExprFunctor::result_type;
  using ExprFunctor::operator();

protected:
  using ExprFunctor::VisitExpr;

private:
  R VisitExpr_(const VarNode *op) override;
  /// @todo (yangjianchao) Supplemet the other visit methods.
  R VisitExprDefault_(const Object *op) override;
};

/// Helpers to transfer Everything to String
template <typename... Args> String CastString(Args... args);

String Everything2String(const std::string &ref) { return String(ref); }
String Everything2String(const int ref) { return String(std::to_string(ref)); }
String Everything2String(const tvm::runtime::ObjectRef &ref) {
  std::ostringstream os;
  tvm::ReprPrinter printer(os);
  printer.Print(ref);
  return String(os.str());
}
String Everything2String(DataType &ref) {
  return CastString("code:", DLDataTypeCode2Str(DLDataTypeCode(ref.code())),
                    "bits:", ref.bits(), "lanes:", ref.lanes());
}

}  // namespace expr_functor_test
