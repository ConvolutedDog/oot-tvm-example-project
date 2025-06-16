#ifndef UTILS_H
#define UTILS_H

#include "test-func-registry.h"
#include <sys/ioctl.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/function.h>
#include <type_traits>
#include <unistd.h>

using tvm::runtime::TVMRetValue;
#include <tvm/../../src/node/attr_registry.h>

using std::string;
using std::vector;
using tvm::runtime::Array;
using tvm::runtime::String;
using tvm::runtime::TVMRetValue;
#include <tvm/../../src/node/attr_registry.h>

/// @brief Checks whether the given module contains any Relax functions.
///
/// This is a templated function that works with either tvm::tir::PrimFunc or
/// tvm::IRModule.
///
/// For PrimFunc inputs, always returns false since they can't contain Relax functions.
/// For IRModule inputs, scans all functions in the module to find Relax functions.
///
/// @tparam T The type of the input module. Must be either tvm::tir::PrimFunc or
///         tvm::IRModule.
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

/// @brief Lists all registered operator names in TVM.
///
/// This function accesses the global operator registry and returns an array containing
/// the names of all registered operators.
///
/// @return Array<String> An array of operator names.
inline Array<String> ListAllOpNames() {
  using OpRegistry = tvm::AttrRegistry<tvm::OpRegEntry, tvm::Op>;
  return OpRegistry::Global()->ListAllNames();
}

/// @brief Retrieves an operator by its string name.
///
/// This function accesses the global operator registry and returns the operator
/// associated with the given string name.
///
/// @param opName The string name of the operator.
/// @return The operator associated with the given string name.
inline const tvm::Op &GetOpByStringName(const String &opName) {
  tvm::OpRegEntry &opregentry = tvm::OpRegEntry::RegisterOrGet(opName);
  return opregentry.op();
}

/// @brief Get the additional attributes' value of the operator by its string name and the
/// attribute name.
///
/// @tparam ValueType The type of the additional attribute's value.
/// @param opName The string name of the operator.
/// @param attrName The string name of the attribute.
/// @return The additional attribute's value.
template <typename ValueType>
ValueType GetAdditioanlAttrValue(const String &opName, const String &attrName) {
  using OpRegistry = tvm::AttrRegistry<tvm::OpRegEntry, tvm::Op>;
  using AttrMapType = tvm::AttrRegistryMapContainerMap<tvm::Op>;
  const AttrMapType &attrMap = OpRegistry::Global()->GetAttrMap(attrName);
  const tvm::Op &op = GetOpByStringName(opName);
  ICHECK(attrMap.count(op)) << "Attribute " << attrName
                            << " has not been registered for operator " << opName;
  return attrMap.operator[](op);
}

/// @brief Get the normal attr value of a relax::Call instance.
///
/// @param expr The relax::Call instance to get the normal attr value from.
/// @return The normal attr value of the relax::Call instance.
inline tvm::Attrs GetNormalAttrValue(const tvm::RelaxExpr &expr) {
  tvm::relax::Call call = tvm::runtime::Downcast<tvm::relax::Call>(expr);
  tvm::Attrs attrs = call->attrs;
  return attrs;
}

/// @brief Adjusts screen printing of a collection of strings for better terminal display.
///
/// This template function formats a collection of strings into multiple columns based on
/// terminal width. It supports different container types (Array<String>, vector<String>,
/// vector<string>) and can optionally sort the items.
///
/// @tparam T The container type, must be one of Array<String>, vector<String>, or
///         vector<string>.
/// @tparam typename SFINAE enable_if parameter to restrict template instantiation.
/// @param os The output stream to write to.
/// @param things The collection of strings to display.
/// @param sort Whether to sort the strings alphabetically before display (default true).
template <typename T, typename = std::enable_if_t<std::is_same_v<T, Array<String>> ||
                                                  std::is_same_v<T, vector<String>> ||
                                                  std::is_same_v<T, vector<string>>>>
void AdjustScreenPrint(std::ostream &os, T things, bool sort = true) {
  // Convert input to vector<string> for uniform handling.
  vector<string> thingsvec = std::vector<string>(things.begin(), things.end());

  if (sort)
    std::sort(thingsvec.begin(), thingsvec.end());

  // Get the width of terminal.
  struct winsize w;
  int terminalWidth = 80;  // Default width if detection fails
  if (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
    terminalWidth = w.ws_col;
  }

  // Calculate maximum name width for column alignment.
  size_t maxNameWidth = 0;
  for (const auto &name : thingsvec) {
    maxNameWidth = std::max(maxNameWidth, name.size());
  }
  maxNameWidth += 2;  // Padding

  size_t columns = std::max(1, static_cast<int>(terminalWidth / maxNameWidth));
  size_t rows = (thingsvec.size() + columns - 1) / columns;

  os << BLUE_TEXT;
  // Print in column-major order.
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < columns; ++col) {
      size_t index = col * rows + row;  // Col-major
      if (index < thingsvec.size()) {
        os << thingsvec[index];
        for (size_t i = 0; i < maxNameWidth - thingsvec[index].size(); ++i)
          os << " ";
      }
    }
    os << "\n";
  }
  os << RESET_TEXT;
}

#endif
