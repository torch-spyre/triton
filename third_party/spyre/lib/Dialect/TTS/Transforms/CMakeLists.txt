# Transforms whose subject is the `tts` dialect's own abstraction -- its marker
# ops.  Peer of Dialect/KTDP/Transforms/ and, like it, named for the dialect it
# is about.
#
# LINK_LIBS derived from actual #includes.  LowerTTSMarkers declares no
# dependentDialects: it constructs no op and no dialect attribute, only builtin
# attributes under `tts.` names.
#
# The edges worth naming:
#   TTSIR      the dialect whose marker ops this consumes, and the attribute
#              names it writes them under.
#   KTDPUtils  isLoweredDescriptor / getDescriptorMemView, which are how a
#              `tts.tensor_layout` operand resolves to its memory view rather
#              than restating that walk.  That is what the KTDP helpers are
#              published for -- neither pass library has to be reachable from
#              the other.
#
# No TritonIR: `tts.tensor_layout`'s `!tt.tensordesc` operand is only ever looked
# at through those two helpers, which speak memref.
add_triton_library(TTSTransforms
  LowerTTSMarkers.cpp

  DEPENDS
  TTSTransformsIncGen
  # Includes the dialect's generated op declarations through
  # Dialect/TTS/IR/Dialect.h, so a clean parallel build must not rely on the
  # link edge to order that tablegen ahead of it.
  TTSIncGen

  LINK_LIBS PUBLIC
  TTSIR
  KTDPUtils
  KTIR::KtdpDialect
  MLIRIR
  MLIRPass
)
