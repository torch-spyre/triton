# TTIR -> KTIR conversions: each of these passes has a source dialect (tt) and a
# target dialect.  Top-level Conversion/, not Dialect/KTDP/Conversion/, to
# follow the rest of the stack -- upstream MLIR has Conversion/XToY and no
# Dialect/*/Conversion*, and this repo already has lib/Conversion/TritonToTritonGPU.
#
# LINK_LIBS is derived from what these files actually #include, plus the
# dependentDialects their Passes.td declares -- tablegen turns those into
# registry.insert<> in the generated pass base, so a pass can need a dialect it
# includes no header for.  MLIRSCFDialect is exactly that case: nothing here
# mentions scf::, but DistributeWork declares mlir::scf::SCFDialect.
#
# Note add_triton_library makes an OBJECT library, so a wrong entry here will
# very often still link.  Treat this list as documentation that has to be right
# rather than as something the build will check.
add_triton_library(TritonToKTIR
  LowerDescriptorMemory.cpp
  LowerScalarLoad.cpp
  LowerComputeOps.cpp
  LowerSpyreOps.cpp
  ConvertFunctions.cpp
  LowerInterTile.cpp
  DistributeWork.cpp
  # Helpers used only by the files above; declared in ConversionUtils.h, which
  # stays out of include/ for that reason.
  ConversionUtils.cpp

  DEPENDS
  TritonToKTIRIncGen
  # LowerDescriptorMemory.cpp and LowerScalarLoad.cpp include the tts dialect's
  # generated op declarations through Dialect/TTS/IR/Dialect.h, so a clean
  # parallel build must not rely on the link edge to order that tablegen ahead
  # of them.
  TTSIncGen

  LINK_LIBS PUBLIC
  KTDPUtils
  SpyreUtils
  # For the tts.tensor_layout op class, which two of these passes name in their
  # conversion target's legal set so the marker survives to lower-tts-markers.
  # Neither reads it and neither writes the attribute.
  TTSIR
  KTIR::KtdpDialect
  KTIR::SpyreOpDialect
  TritonIR
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  # No scf:: anywhere in these files; DistributeWork's dependentDialects needs it.
  MLIRSCFDialect
  # mlir/Interfaces/FunctionInterfaces.h, used directly by ConvertFunctions and
  # DistributeWork rather than reached through MLIRFuncDialect.
  MLIRFunctionInterfaces
  # applyPartialConversion and DialectConversion.cpp.
  MLIRTransformUtils
  # mlir/Interfaces/SideEffectInterfaces.h, for isOpTriviallyDead in
  # ConversionUtils.cpp.
  MLIRSideEffectInterfaces
)
