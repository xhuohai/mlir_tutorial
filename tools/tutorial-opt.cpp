#include "../lib/Transform/Affine/Passes.h"
#include "../lib/Transform/Arith/Passes.h"
#include "../lib/Dialect/Poly/PolyDialect.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
// #include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                    mlir::tutorial::poly::PolyDialect>();

    mlir::affine::registerAffinePasses();
    mlir::tutorial::registerArithPasses();
    mlir::tutorial::registerAffinePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}