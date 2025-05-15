#ifndef DEPRECATED_H
#define DEPRECATED_H

#include "ir/analysis-test.h"
#include "ir/attrs-test.h"
#include "ir/diagnostic-test.h"
#include "ir/expr-test.h"
#include "ir/function-test.h"
#include "ir/global-info-test.h"
#include "ir/global-var-supply-test.h"
#include "ir/module-test.h"
#include "ir/name-supply-test.h"
#include "ir/op-test.h"
#include "ir/pass-test.h"
#include "ir/replace-global-vars-test.h"
#include "ir/source-map-test.h"
#include "ir/transform-test.h"
#include "ir/type-functor-test.h"
#include "ir/type-test.h"
#include "node/functor-test.h"
#include "node/reflection-test.h"
#include "relax/expr-test.h"
#include "runtime/inplacearraybase-test.h"
#include "runtime/ndarrayutils-test.h"
#include "runtime/object-test.h"
#include "runtime/tvmpodvalue-test.h"
#include "target/target-kind-test.h"
#include "target/target-test.h"
#include "tir/var-test.h"

namespace deprecated {

[[deprecated("Deprecated and will not be maintained, please use TestMethod3() instead.")]]
void TestMethod2();

}  // namespace deprecated

#endif
