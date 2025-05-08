#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/packed_func.h"

namespace tvmpodvalue_test {

using tvm::runtime::TVMPODValue_;

class TVMPODValueDerived : public tvm::runtime::TVMPODValue_ {
public:
  TVMPODValueDerived(TVMValue value, int typecode) : TVMPODValue_(value, typecode) {}
};

void TvmPodValueTest();

}  // namespace tvmpodvalue_test

void TvmPodValueTest();
