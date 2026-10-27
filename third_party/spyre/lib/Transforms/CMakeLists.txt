# Passes whose subject is upstream structure or the whole program rather than
# any one dialect's abstractions: UnaliasLinalgOuts, DropReductionInitFill,
# CanonicalizeGenericOperands and FuseGenerics act on linalg form -- generic
# bodies, indexing maps, reshape reassociations -- NormalizeForDevice hosts
# patterns rewriting upstream ops the toolchain downstream of this tree will not
# take, wherever they appear, and MaterializeBaseAddresses acts on the entry
# function signature.  Sitting at the top level does not stop them including
# dialect headers.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# One absence is the interesting content here: no TritonIR, because none of the
# passes touches a tt op.  MLIRTransformUtils used to be a second such absence,
# every pass here walking and rewriting directly; CanonicalizeGenericOperands,
# FuseGenerics and NormalizeForDevice drive the greedy rewriter now, so the
# claim is only about the dialects, not about the rewrite machinery.
add_triton_library(SpyreTransforms
  UnaliasLinalgOuts.cpp
  NormalizeForDevice.cpp
  DropReductionInitFill.cpp
  CanonicalizeGenericOperands.cpp
  FuseGenerics.cpp
  LowerSpyreOps.cpp
  MaterializeBaseAddresses.cpp

  DEPENDS
  SpyreTransformsIncGen

  LINK_LIBS PUBLIC
  # For createEmptyTensor in UnaliasLinalgOuts, and GenericOperands.
  SpyreUtils
  # LowerSpyreOps emits spyreop intrinsics -- the one dialect here that is a
  # TARGET rather than a subject.  Its rules match arith and math; it is KTIR ->
  # KTIR and reads no tt op, which is why it is here rather than in Conversion/.
  KTIR::SpyreOpDialect
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  # math.sqrt/exp/rsqrt, which LowerSpyreOps selects intrinsics for.
  MLIRMathDialect
  MLIRTensorDialect
  # For fuseElementwiseOps and deduplicateOperandsAndRemoveDeadResults, which
  # FuseGenerics' GroupComputes rewrites with.
  MLIRLinalgTransforms
  # For applyPatternsGreedily, which drives the patterns of
  # CanonicalizeGenericOperands, FuseGenerics and NormalizeForDevice.
  MLIRTransformUtils
)
