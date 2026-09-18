get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(spyre-triton-opt spyre-triton-opt.cpp)
target_compile_options(spyre-triton-opt PRIVATE -fno-exceptions -fno-rtti)
# SpyreRegisterEverything is the only Spyre library named here: it carries the
# pass libraries and the KTIR dialects behind its own LINK_LIBS, so adding a pass
# library does not reach this file.
target_link_libraries(spyre-triton-opt PRIVATE
  ${triton_libs}
  SpyreRegisterEverything
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIRTransforms
)
