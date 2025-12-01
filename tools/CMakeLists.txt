cmake_minimum_required(VERSION 3.20.0)
project(tutorial-opt)

add_executable(tutorial-opt tutorial-opt.cpp)

target_link_libraries(tutorial-opt PRIVATE
  AffineFullUnroll
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRParser
  MLIROptLib
  MLIRFuncDialect
  MLIRAffineDialect
)