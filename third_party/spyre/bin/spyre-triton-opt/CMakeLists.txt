get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(spyre-triton-opt spyre-triton-opt.cpp)
target_compile_options(spyre-triton-opt PRIVATE -fno-exceptions -fno-rtti)
target_link_libraries(spyre-triton-opt PRIVATE
  ${triton_libs}
  KTDPTransforms
  KTIR::KtdpDialect
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIRTransforms
)
