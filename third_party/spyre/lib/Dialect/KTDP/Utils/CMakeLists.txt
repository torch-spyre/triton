# Leaf library: helpers shared by the TritonToKTIR conversions, the KTDP
# transforms and the top-level Spyre transforms.  It depends on none of them,
# which is precisely what lets those libraries be peers -- were Utility.cpp
# still sitting in one of them, the others would have to depend on it.
#
# LINK_LIBS is derived from what this one file (and its header) include.
add_triton_library(KTDPUtils
  Utility.cpp

  LINK_LIBS PUBLIC
  KTIR::KtdpDialect
  MLIRIR
  MLIRArithDialect
  MLIRTensorDialect
  # mlir/Interfaces/SideEffectInterfaces.h, for isMemoryEffectFree.
  MLIRSideEffectInterfaces
)
