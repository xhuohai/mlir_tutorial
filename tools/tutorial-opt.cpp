#include "lib/Dialect/Poly/PolyDialect.h"
#include "../lib/Transform/Affine/Passes.h"
#include "../lib/Transform/Arith/Passes.h"
#include "lib/Conversion/PolyToStandard/PolyToStandard.h"
#include "lib/Dialect/Poly/PolyDialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllDialects.h"
// #include "mlir/InitAllPasses.h"
// #include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // manager.addPass(mlir::createLowerAffinePass());
  
  // Needed to lower memref.subview
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  
  manager.addPass(mlir::createConvertSCFToCFPass());

  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::complex::ComplexDialect, mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::tutorial::poly::PolyDialect>();

  mlir::registerCanonicalizerPass();
  mlir::registerSCCPPass();
  mlir::registerCSEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerControlFlowSinkPass();
  mlir::tutorial::registerArithPasses();
  mlir::tutorial::registerAffinePasses();

  mlir::tutorial::poly::registerPolyToStandardPass();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::PassPipelineRegistration<>(
      "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
      polyToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}