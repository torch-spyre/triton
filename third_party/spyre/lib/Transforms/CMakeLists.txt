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
  # For Utility.h in UnaliasLinalgOuts; also carries the ktdp include path it
  # needs, so KTIR::KtdpDialect is not named again here.
  KTDPUtils
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
)
