#include "tvm/ir/expr.h"
#include "tvm/ir/module.h"
#include "tvm/ir/type.h"
#include "tvm/node/repr_printer.h"
#include "tvm/runtime/container/map.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/function.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/var.h"
#include <tvm/relax/transform.h>

namespace repr_printer_test {

using tvm::AsLegacyRepr;
using tvm::Dump;
using tvm::ReprLegacyPrinter;
using tvm::ReprPrinter;

using tvm::DataType;
using tvm::GlobalVar;
using tvm::IRModule;
using tvm::PrimExpr;

using tvm::tir::Evaluate;
using tvm::tir::LetStmt;
using tvm::tir::PrimFunc;
using tvm::tir::Stmt;
using tvm::tir::Var;

void NodeAsLegacyReprTest();
void NodeReprPrinterTest();
void NodeReprLegacyPrinterTest();
void NodeDumpTest();

}  // namespace repr_printer_test
