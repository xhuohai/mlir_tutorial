
add_mlir_pdll_library(MulToAddPdllIncGen
  MulToAdd.pdll
  MulToAddPdll.h.inc
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(ArithPasses.h.inc -gen-pass-decls -name=Arith)
mlir_tablegen(ArithPasses.md -gen-pass-doc)
add_public_tablegen_target(MLIRArithTransformsIncGen)
# add_custom_target(header DEPENDS MLIRArithTransformsIncGen)

add_mlir_library(MulToAdd MulToAdd.cpp MulToAddPdll.cpp DEPENDS MulToAddPdllIncGen)
# target_link_libraries(MulToAdd PUBLIC MLIRAffineUtils)
target_include_directories(MulToAdd PUBLIC ${CMAKE_CURRENT_BINARY_DIR})