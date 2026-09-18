# Leaf library, and the only one of ours that links none of ours -- these
# helpers read and build upstream arith/tensor ops and nothing else, which is
# what keeps every pass library free to depend on it.  Contrast KTDPUtils, whose
# helpers build ktdp ops and so carry KTIR::KtdpDialect.
#
# LINK_LIBS is derived from what this one file (and its header) include.
add_triton_library(SpyreUtils
  Utility.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  # arith.constant, read by getConstantInt.
  MLIRArithDialect
  # tensor.empty and tensor.dim, built by createEmptyTensor.
  MLIRTensorDialect
)
