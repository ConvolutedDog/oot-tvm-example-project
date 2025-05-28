#ifndef UTILS_H
#define UTILS_H

#include "tvm/tir/function.h"
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <type_traits>

/// @brief Checks whether the given module contains any Relax functions.
///
/// This is a templated function that works with either tvm::tir::PrimFunc or
/// tvm::IRModule.
///
/// For PrimFunc inputs, always returns false since they can't contain Relax functions.
/// For IRModule inputs, scans all functions in the module to find Relax functions.
///
/// @tparam T The type of the input module. Must be either tvm::tir::PrimFunc or tvm::IRModule.
/// @param mod The input module to check.
/// @return True if the module contains any Relax functions, false otherwise.
template <typename T, typename = std::enable_if_t<std::is_same_v<T, tvm::tir::PrimFunc> ||
                                                  std::is_same_v<T, tvm::IRModule>>>
bool _contains_relax(T mod) {  // NOLINT
  if constexpr (std::is_same_v<T, tvm::tir::PrimFunc>)
    return false;
  if constexpr (std::is_same_v<T, tvm::IRModule>) {
    for (auto &[gvar, func] : mod->functions) {
      if (func.defined() && func->template IsInstance<tvm::relax::FunctionNode>()) {
        return true;
      }
    }
  }
  return false;
}

#endif
