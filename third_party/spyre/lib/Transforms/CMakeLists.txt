# Passes whose subject is upstream structure or the whole program rather than
# any one dialect's abstractions: UnaliasLinalgOuts and DropReductionInitFill
# act on linalg form, MaterializeBaseAddresses on the entry function signature.
# Sitting at the top level does not stop them including dialect headers.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# Two absences are the interesting content here: no TritonIR, because none of
# the three touches a tt op; and no MLIRTransformUtils, because none of them
# drives the conversion framework or the greedy rewriter -- they walk and
# rewrite directly.
add_triton_library(SpyreTransforms
  UnaliasLinalgOuts.cpp
  DropReductionInitFill.cpp
  MaterializeBaseAddresses.cpp

  DEPENDS
  SpyreTransformsIncGen

  LINK_LIBS PUBLIC
  # For createEmptyTensor in UnaliasLinalgOuts.  Not KTDPUtils: none of these
  # three touches a ktdp op, so this library no longer reaches the ktdp dialect
  # at all -- the third absence worth noting alongside the two below.
  SpyreUtils
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
)
