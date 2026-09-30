# The tts dialect: the first dialect defined in this tree rather than inherited
# from ktir-mlir-frontend or upstream.  It defines no types and one op -- what it
# owns is the `tts.tensor_layout` attribute name, the verifier hook that name
# routes to, the authoring op of the same name, and the one structural checker
# all of them share.
#
# LINK_LIBS derived from what Dialect.cpp, Ops.cpp and the shared header include.
# KTIR::KtdpDialect is for the one op the attribute verifier gates on: the
# attribute is rejected anywhere other than a ktdp.construct_memory_view, and the
# view's result memref is where the logical rank is measured.  That edge points at
# a dialect and not at a pass library, so no peer of this one has to be reachable
# from it -- which is what lets KTDPTransforms depend on this rather than the
# other way round.
#
# TritonIR is for TensorLayoutOp's `!tt.tensordesc` operand, and only points that
# way: Triton names nothing of ours.
add_triton_library(TTSIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TTSIncGen

  LINK_LIBS PUBLIC
  KTIR::KtdpDialect
  TritonIR
  MLIRIR
)
