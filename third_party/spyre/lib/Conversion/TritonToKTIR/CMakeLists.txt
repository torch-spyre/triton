# TTIR -> KTIR conversions: each of these passes has a source dialect (tt) and a
# target dialect.  Top-level Conversion/, not Dialect/KTDP/Conversion/, to
# follow the rest of the stack -- upstream MLIR has Conversion/XToY and no
# Dialect/*/Conversion*, and this repo already has lib/Conversion/TritonToTritonGPU.
#
# LINK_LIBS is derived from what these seven files actually #include, plus the
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

  DEPENDS
  TritonToKTIRIncGen

  LINK_LIBS PUBLIC
  KTDPUtils
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
)
