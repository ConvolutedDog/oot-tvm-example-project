#include "tvm/ir/transform.h"

namespace transform_test {

using tvm::transform::PassContext;
using tvm::transform::PassInfo;
using tvm::transform::Pass;
using tvm::transform::Sequential;
using tvm::transform::CreateModulePass;
using tvm::transform::ApplyPassToFunction;
using tvm::transform::PrintIR;

}
