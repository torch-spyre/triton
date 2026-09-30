# Leaf library, and the only one of ours that links no dialect of OURS -- these
# helpers read and build upstream ops only, which is what keeps every pass
# library free to depend on it.  Contrast KTDPUtils, whose helpers build ktdp ops
# and so carry KTIR::KtdpDialect.
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
  # tt.make_tensor_descriptor, read by getDescriptorLogicalLayout. Upstream
  # Triton's dialect, so it does not cost this library its leaf position: every
  # pass library that links SpyreUtils already sits above TritonIR.
  TritonIR
)
