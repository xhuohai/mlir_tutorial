#include "../lib/Transform/Affine/AffineFullUnroll.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::affine::AffineDialect>();

    mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
    mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}