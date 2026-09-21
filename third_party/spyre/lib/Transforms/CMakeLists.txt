# Passes whose subject is upstream structure or the whole program rather than
# any one dialect's abstractions: UnaliasLinalgOuts, DropReductionInitFill and
# FoldDataMovementGenerics act on linalg form -- generic bodies, indexing maps,
# reshape reassociations -- and MaterializeBaseAddresses on the entry function
# signature.  Sitting at the top level does not stop them including dialect
# headers.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# One absence is the interesting content here: no TritonIR, because none of the
# four touches a tt op.  MLIRTransformUtils used to be a second such absence,
# every pass here walking and rewriting directly; FoldDataMovementGenerics ended
# that by driving the greedy rewriter, so the claim now is only about the
# dialects, not about the rewrite machinery.
add_triton_library(SpyreTransforms
  UnaliasLinalgOuts.cpp
  DropReductionInitFill.cpp
  FoldDataMovementGenerics.cpp
  MaterializeBaseAddresses.cpp

  DEPENDS
  SpyreTransformsIncGen
  # FoldDataMovementGenerics includes the tts dialect's generated declarations
  # through Dialect/TTS/IR/Dialect.h, so a clean parallel build must not rely on
  # the link edge to order that tablegen ahead of it.
  TTSIncGen

  LINK_LIBS PUBLIC
  # For createEmptyTensor in UnaliasLinalgOuts.
  SpyreUtils
  # FoldDataMovementGenerics refuses an unabsorbable re-indexing op only on a
  # path RewriteDescriptorLayoutGeneric will physicalize, and asks that question
  # with that pass's own seed traversals -- hence the ktdp dialect, KTDPUtils
  # where the shared walks live, and TTSIR for the layout attribute's name.  It
  # is the only pass here that reaches a ktdp op; the other three still do not.
  KTDPUtils
  TTSIR
  KTIR::KtdpDialect
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  # For populateElementwiseOpsFusionPatterns, the composition
  # FoldDataMovementGenerics supplies a control function to.
  MLIRLinalgTransforms
  # For applyPatternsGreedily, which the same pass drives its two pattern kinds
  # with.
  MLIRTransformUtils
)
