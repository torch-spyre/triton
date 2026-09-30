# Transforms whose subject is the KTDP dialect's own abstractions -- memory
# views and access tiles.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# No func, math, memref or spyreop: this pass touches none of them.
add_triton_library(KTDPTransforms
  RewriteDescriptorLayoutGeneric.cpp

  DEPENDS
  KTDPTransformsIncGen
  # RewriteDescriptorLayoutGeneric.cpp includes the tts dialect's generated
  # declarations through Dialect/TTS/IR/Dialect.h, so a clean parallel build
  # must not rely on the link edge to order that tablegen ahead of it.
  TTSIncGen

  LINK_LIBS PUBLIC
  KTDPUtils
  SpyreUtils
  # For the shared tts.tensor_layout checker, which this library's generic
  # layout pass and the dialect's own verifier both call.
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
