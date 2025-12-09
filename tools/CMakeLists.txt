add_executable(tutorial-opt tutorial-opt.cpp)

target_link_libraries(tutorial-opt PRIVATE
  AffineFullUnroll
  MulToAdd
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRParser
  MLIROptLib
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  PolyDialect
  PolyToStandard
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLinalgTransforms
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRSCFToControlFlow
  MLIRTensorToLinalg
  MLIRReconcileUnrealizedCasts
  MLIRAffineTransforms
  MLIRMemRefToLLVM
)

if(NOT LLVM_ENABLE_RTTI)
  llvm_update_compile_flags(tutorial-opt) # 针对你的 target
endif()

install(TARGETS tutorial-opt DESTINATION bin)