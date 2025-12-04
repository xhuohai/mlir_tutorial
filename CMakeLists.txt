cmake_minimum_required(VERSION 3.20.0)
project(mlir-tutorial)

set(CMAKE_CXX_STANDARD 17)

# --------------------------------------------------------------------------
# 1. 寻找并加载 LLVM/MLIR
# --------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

# 将 MLIR 的 CMake 模块路径加入到当前路径，以便使用 add_mlir_tool 等辅助函数
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})

# 这一步是为了确保使用和 LLVM 相同的编译定义 (如 RTTI 设置)
add_definitions(${LLVM_DEFINITIONS})

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(lib/Dialect/Poly)
add_subdirectory(lib/Transform)
add_subdirectory(tools)
