# Leaf library, and the only one of ours that links no dialect of OURS -- these
# helpers read and build upstream ops only, which is what keeps every pass
# library free to depend on it.  Contrast KTDPUtils, whose helpers build ktdp ops
# and so carry KTIR::KtdpDialect.
#
# LINK_LIBS is derived from what these files (and their headers) include.
add_triton_library(SpyreUtils
  Utility.cpp
  GenericOperands.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  # arith.constant, read by getConstantInt.
  MLIRArithDialect
  # tensor.empty and tensor.dim, built by createEmptyTensor.
  MLIRTensorDialect
  # linalg.generic, whose operands GenericOperands checks and rewrites.
  MLIRLinalgDialect
  # tt.make_tensor_descriptor, read by getDescriptorLogicalLayout. Upstream
  # Triton's dialect, so it does not cost this library its leaf position: every
  # pass library that links SpyreUtils already sits above TritonIR.
  TritonIR
)
