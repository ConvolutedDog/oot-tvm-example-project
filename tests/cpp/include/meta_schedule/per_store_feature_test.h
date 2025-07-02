#include "test-func-registry.h"
#include <tvm/arith/analyzer.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/var.h>
#include <tvm/tir/expr_functor.h>
#include <vector>

namespace per_store_feature_test {
using tvm::PrimExpr;
using tvm::arith::Analyzer;
using tvm::tir::Var;
using tvm::tir::ExprVisitor;
using tvm::tir::MulNode;
using tvm::tir::AddNode;
using tvm::tir::IntImmNode;
using tvm::tir::VarNode;
using MultiIndex = std::vector<PrimExpr>;
using IntVec = std::vector<long>;

void RelaxAndUnionTest();

/// @attention This function is copied from the `RelaxAndUnion` function in the
/// `per_store_feature.cc` file， because the `RelaxAndUnion` function is not exported in
/// the head file and will trigger the register error if we just simply include the
/// `per_store_feature.cc` file.
/*!
 * \brief Relax each of the multi-indexing pattern according to the domains bound in the
 * analyzer, and then union them into a single region
 * \param multi_index_pattern A list of multi-index pattern to be relaxed
 * \param numel The size of the single region after union
 * \param analyzer The analyzer that contains the domain information
 * \return The relaxed and unioned region
 * \note This function computes the size of the region that every dimension covers.
 * For example, if the multi-index pattern is {{0,1}, {1,2}, {3, 3}}
 * The access_shape will be {4, 3} which means the first dimension covers 4
 * elements(0,1,2,3) and the second dimension covers 3 elements(1,2,3).
 * \note This function is used to compute the access shape of the buffer.
 */
IntVec RelaxAndUnion(const std::vector<MultiIndex> &multi_indices, int64_t *numel,
                     tvm::arith::Analyzer *analyzer) {
  *numel = 1;
  if (multi_indices.empty()) {
    return {};
  }
  int n_indices = multi_indices.size();
  int ndim = multi_indices[0].size();
  IntVec access_shape(ndim, 0);
  for (int i = 0; i < ndim; ++i) {
    int64_t minimum = tvm::arith::ConstIntBound::kPosInf;
    int64_t maximum = tvm::arith::ConstIntBound::kNegInf;
    for (int j = 0; j < n_indices; ++j) {
      tvm::arith::ConstIntBound bound = analyzer->const_int_bound(multi_indices[j][i]);
      minimum = std::min(minimum, bound->min_value);
      maximum = std::max(maximum, bound->max_value);
    }
    *numel *= maximum - minimum + 1;
    access_shape[i] = maximum - minimum + 1;
  }
  return access_shape;
}

/*!
 * \brief Given a list of multi-index pattern, return the minimal stride of a variable on it
 * \param multi_indices The list of multi-index pattern
 * \param buffer_stride The stride of the buffer
 * \param var The variable to be checked
 * \return The minimal stride of the variable on the multi-index pattern
 */
int64_t GetVarStride(const std::vector<MultiIndex>& multi_indices, const IntVec& buffer_stride,
                     const Var& var) {
  class CoefficientExtractor : private ExprVisitor {
   public:
    static int64_t Extract(const PrimExpr& expr, const Var& var) {
      CoefficientExtractor extractor(var);
      extractor.VisitExpr(expr);
      return (extractor.visited_var && !extractor.visited_mul && !extractor.visited_add)
                 ? 1
                 : (extractor.visited_var ? extractor.stride : 0);
    }

   private:
    explicit CoefficientExtractor(const Var& var)
        : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

    void VisitExpr_(const MulNode* node) override {
      ExprVisitor::VisitExpr_(node);
      if (visited_var && !visited_add) {
        if (const auto* a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (const auto* b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }

    void VisitExpr_(const AddNode* node) override {
      ExprVisitor::VisitExpr_(node);
      if (visited_var && !visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }

    void VisitExpr_(const VarNode* node) override {
      if (node == var.get()) {
        visited_var = true;
        stride = 2;
      }
    }

    const Var& var;
    int64_t stride;
    bool visited_var;
    bool visited_add;
    bool visited_mul;
  };

  constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
  int ndim = buffer_stride.size();
  // Calculate the min stride possible
  int64_t result = kNotFound;
  for (const MultiIndex& multi_index : multi_indices) {
    ICHECK_EQ(multi_index.size(), buffer_stride.size());
    // Find the rightest dimension that contains the given variable
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t coef = CoefficientExtractor::Extract(multi_index[i], var);
      if (coef != 0) {
        result = std::min(result, std::abs(coef) * buffer_stride[i]);
        break;
      }
    }
  }
  return (result == kNotFound) ? 0 : result;
}

}  // namespace per_store_feature_test