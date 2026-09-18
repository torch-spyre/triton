# Leaf library: the ktdp-building helpers shared by the TritonToKTIR
# conversions and the KTDP transforms.  It depends on neither of them, which is
# precisely what lets those libraries be peers -- were Utility.cpp still sitting
# in one of them, the other would have to depend on it.
#
# Everything left here builds or inspects ktdp ops, so KTIR::KtdpDialect is a
# real dependency and the library's name is accurate.  Helpers that are about no
# dialect are in SpyreUtils instead.
#
# LINK_LIBS is derived from what this one file (and its header) include.
add_triton_library(KTDPUtils
  Utility.cpp

  LINK_LIBS PUBLIC
  KTIR::KtdpDialect
  MLIRIR
  # arith.index_cast, built by buildAccessTile.
  MLIRArithDialect
)
