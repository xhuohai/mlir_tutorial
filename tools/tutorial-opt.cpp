#include "../lib/Transform/Affine/Passes.h"
#include "../lib/Transform/Arith/Passes.h"
#include "../lib/Dialect/Poly/PolyDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
// #include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::affine::AffineDialect, mlir::scf::SCFDialect,
                    mlir::tutorial::poly::PolyDialect>();

    mlir::registerCSEPass();
    mlir::registerLoopInvariantCodeMotionPass();
    mlir::registerControlFlowSinkPass();
    mlir::tutorial::registerArithPasses();
    mlir::tutorial::registerAffinePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}