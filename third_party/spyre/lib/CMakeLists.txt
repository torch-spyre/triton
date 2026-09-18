add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Transforms)

# Sits above the pass libraries and is depended on by no other library, which is
# what lets it link all of them. Keep in step with RegisterEverything.cpp: a new
# pass library added there needs its target added here.
add_triton_library(SpyreRegisterEverything
  RegisterEverything.cpp

  # Named directly rather than left to the pass libraries' own DEPENDS: this
  # file includes all three generated Passes.h.inc, so a clean parallel build
  # must not rely on the link edge to order tablegen ahead of it.
  DEPENDS
  TritonToKTIRIncGen
  KTDPTransformsIncGen
  SpyreTransformsIncGen

  LINK_LIBS PUBLIC
  TritonToKTIR
  KTDPTransforms
  SpyreTransforms
  KTIR::KtdpDialect
  KTIR::SpyreOpDialect
  MLIRIR
  # The generated registration code this file inlines builds PassRegistration
  # objects, so the dependency is real and not just the Passes.h include path.
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
)
