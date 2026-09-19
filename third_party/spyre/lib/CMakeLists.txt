add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(Utils)

# The pass sequences each compile stage runs, spelled once. Sits above the pass
# libraries because it names their create* factories, and below
# SpyreRegisterEverything, which registers the sequences as CLI pipelines.
#
# LINK_LIBS derived from what Pipeline.cpp includes. MLIRTransforms is for
# canonicalize and CSE, MLIRLinalgTransforms for
# createConvertElementwiseToLinalgPass -- the two upstream passes the sequences
# reach for, and the reason this is not just a list of our own factories.
add_triton_library(SpyrePipeline
  Pipeline.cpp

  # This file includes all three generated Passes.h.inc, one of them for the
  # RewriteDescriptorLayout options struct, so a clean parallel build must not
  # rely on the link edge to order tablegen ahead of it.
  DEPENDS
  TritonToKTIRIncGen
  KTDPTransformsIncGen
  SpyreTransformsIncGen

  LINK_LIBS PUBLIC
  TritonToKTIR
  KTDPTransforms
  SpyreTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
)

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
  # For the two stage pipelines this file registers as CLI flags.
  SpyrePipeline
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
