#include "../include/object-test.h"

using namespace object_test;

/// @brief This macros actually calls `::_GetOrAllocRuntimeTypeIndex()`
/// function, this function will calculate a static variable `tindex`.
/// This `tindex` will be returned during the call of function
/// `RuntimeTypeIndex()` which will allocate the runtime `type_index_`
/// for the node that inherits from `Object`. This macro will define a
/// global variable but this variable will never be used, its function
/// is to call this macro to initialize the static `tindex` variable.
/// @note This is not necessary, because it will be called during the
/// initialization of each node if you didn't call this macro here.
TVM_REGISTER_OBJECT_TYPE(TestCanDerivedFromObject);
TVM_REGISTER_OBJECT_TYPE(TestDerived1);
TVM_REGISTER_OBJECT_TYPE(TestDerived2);
TVM_REGISTER_OBJECT_TYPE(TestFinalObject);
