set(LLVM_TARGET_DEFINITIONS PolyToStandard.td)
mlir_tablegen(PolyToStandard.h.inc -gen-pass-decls -name=PolyToStandard)
add_public_tablegen_target(MLIRPolyToStandardIncGen)

add_mlir_library(PolyToStandard PolyToStandard.cpp)
target_include_directories(PolyToStandard PUBLIC ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_link_libraries(PolyToStandard PUBLIC PolyDialect MLIRFuncTransforms MLIRTransforms)