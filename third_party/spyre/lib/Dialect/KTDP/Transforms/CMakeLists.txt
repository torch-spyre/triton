# Transforms whose subject is the KTDP dialect's own abstractions -- memory
# views and access tiles.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# No func, math, memref or spyreop: no pass here touches them.
add_triton_library(KTDPTransforms
  MaterializePinnedBuffers.cpp
  RewriteDescriptorLayoutGeneric.cpp

  DEPENDS
  KTDPTransformsIncGen
  # RewriteDescriptorLayoutGeneric.cpp and MaterializePinnedBuffers.cpp include
  # the tts dialect's generated declarations through Dialect/TTS/IR/Dialect.h, so
  # a clean parallel build must not rely on the link edge to order that tablegen
  # ahead of them.
  TTSIncGen

  LINK_LIBS PUBLIC
  KTDPUtils
  SpyreUtils
  # For the shared tts.tensor_layout checker, which this library's generic
  # layout pass and the dialect's own verifier both call, and for the two
  # tts attribute names MaterializePinnedBuffers reads.
  TTSIR
  KTIR::KtdpDialect
  TritonIR
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRTensorDialect
  # GreedyPatternRewriteDriver.cpp.
  MLIRTransformUtils
)
