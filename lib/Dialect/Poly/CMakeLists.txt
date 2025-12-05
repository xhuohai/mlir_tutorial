set(LLVM_TARGET_DEFINITIONS PolyDialect.td)
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS PolyTypes.td)
mlir_tablegen(PolyTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PolyTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS PolyOps.td)
mlir_tablegen(PolyOps.h.inc -gen-op-decls)
mlir_tablegen(PolyOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS PolyPatterns.td)
mlir_tablegen(PolyCanonicalize.cpp.inc -gen-rewriters)

add_public_tablegen_target(MLIRPolyDialectIncGen)
# add_custom_target(header DEPENDS MLIRPolyDialectIncGen)

add_mlir_library(PolyDialect PolyDialect.cpp PolyOps.cpp)
target_include_directories(PolyDialect PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(PolyDialect PUBLIC MLIRRewrite)