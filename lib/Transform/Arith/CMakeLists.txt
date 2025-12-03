
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(ArithPasses.h.inc -gen-pass-decls -name=Arith)
add_public_tablegen_target(MLIRArithTransformsIncGen)
# add_custom_target(header DEPENDS MLIRArithTransformsIncGen)

add_mlir_library(MulToAdd MulToAdd.cpp)
# target_link_libraries(MulToAdd PUBLIC MLIRAffineUtils)
target_include_directories(MulToAdd PUBLIC ${CMAKE_CURRENT_BINARY_DIR})