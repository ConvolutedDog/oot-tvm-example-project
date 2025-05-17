#include "runtime/tvmpodvalue-test.h"
#include "test-func-registry.h"

namespace tvmpodvalue_test {

void RuntimeTvmPodValueTest() {
  LOG_SPLIT_LINE("RuntimeTvmPodValueTest");

  /// typedef union {
  ///   int64_t v_int64;
  ///   double v_float64;
  ///   void* v_handle;
  ///   const char* v_str;
  ///   DLDataType v_type;
  ///   DLDevice v_device;
  /// } TVMValue;
  TVMValue value{0x333};

  /// typedef enum {
  ///   kTVMArgInt = kDLInt,
  ///   kTVMArgFloat = kDLFloat,
  ///   kTVMOpaqueHandle = 3U,
  ///   kTVMNullptr = 4U,
  ///   kTVMDataType = 5U,
  ///   kDLDevice = 6U,
  ///   kTVMDLTensorHandle = 7U,
  ///   kTVMObjectHandle = 8U,
  ///   kTVMModuleHandle = 9U,
  ///   kTVMPackedFuncHandle = 10U,
  ///   kTVMStr = 11U,
  ///   kTVMBytes = 12U,
  ///   kTVMNDArrayHandle = 13U,
  ///   kTVMObjectRValueRefArg = 14U,
  ///   kTVMArgBool = 15U,
  ///   // Extension codes for other frameworks to integrate TVM PackedFunc.
  ///   // To make sure each framework's id do not conflict, use first and
  ///   // last sections to mark ranges.
  ///   // Open an issue at the repo if you need a section of code.
  ///   kTVMExtBegin = 16U,
  ///   kTVMNNVMFirst = 16U,
  ///   kTVMNNVMLast = 20U,
  ///   // The following section of code is used for non-reserved types.
  ///   kTVMExtReserveEnd = 64U,
  ///   kTVMExtEnd = 128U,
  /// } TVMArgTypeCode;
  {
    int typecode = TVMArgTypeCode::kTVMNullptr;
    TVMPODValueDerived derived{value, typecode};
    LOG_PRINT_VAR((void *)derived);
  }
  {
    int typecode = TVMArgTypeCode::kTVMDLTensorHandle;
    TVMPODValueDerived derived{value, typecode};
    LOG_PRINT_VAR((void *)derived);
  }
}

}  // namespace tvmpodvalue_test

REGISTER_TEST_SUITE(tvmpodvalue_test::RuntimeTvmPodValueTest,
                    runtime_tvmpodvalue_test_RuntimeTvmPodValueTest);
