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

  LINK_LIBS PUBLIC
  # For createEmptyTensor in UnaliasLinalgOuts.  Not KTDPUtils: none of these
  # four touches a ktdp op, so this library no longer reaches the ktdp dialect
  # at all -- the second absence worth noting alongside the one above.
  SpyreUtils
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
