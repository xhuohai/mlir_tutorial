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
  MLIRPass
  MLIRAffineTransforms
  PolyDialect
)

install(TARGETS tutorial-opt DESTINATION bin)