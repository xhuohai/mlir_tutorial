#ifndef LIB_TRANSFORM_ARITH_PASSES_H_
#define LIB_TRANSFORM_ARITH_PASSES_H_

#include "MulToAdd.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_REGISTRATION
#include "ArithPasses.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_ARITH_PASSES_H_