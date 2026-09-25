# Transforms whose subject is the `tts` dialect's own abstraction -- its marker
# ops.  Peer of Dialect/KTDP/Transforms/ and, like it, named for the dialect it
# is about.
#
# LINK_LIBS derived from actual #includes.  LowerTTSMarkers declares no
# dependentDialects, because it constructs no op and no dialect attribute;
# PlacePinnedValues declares two, because it constructs ktdp ops and an arith
# cast.
#
# The edges worth naming:
#   TTSIR      the dialect whose ops these consume, the attribute names one of
#              them writes under, and `matchPinAddress`, which PlacePinnedValues
#              shares with the pin's own verifier.
#   KTDPUtils  isLoweredDescriptor / getDescriptorMemView, which are how a
#              `tts.tensor_layout` operand resolves to its memory view, and
#              buildMemoryView / buildAccessTile, which are how PlacePinnedValues
#              builds a pinned value's buffer rather than restating either.  That
#              is what the KTDP helpers are published for -- neither pass library
#              has to be reachable from the other.
#
# No TritonIR: `tts.tensor_layout`'s `!tt.tensordesc` operand is only ever looked
# at through those two helpers, which speak memref, and the one Triton op a pinned
# address may name is matched inside TTSIR.
add_triton_library(TTSTransforms
  LowerTTSMarkers.cpp
  PlacePinnedValues.cpp

  DEPENDS
  TTSTransformsIncGen
  # Both files include the dialect's generated op declarations through
  # Dialect/TTS/IR/Dialect.h, so a clean parallel build must not rely on the
  # link edge to order that tablegen ahead of them.
  TTSIncGen

  LINK_LIBS PUBLIC
  TTSIR
  KTDPUtils
  KTIR::KtdpDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
)
