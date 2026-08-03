add_triton_library(KTDPTransforms
  ConvertFunctions.cpp
  LowerDescriptorMemory.cpp
  LowerScalarLoad.cpp
  LowerComputeOps.cpp
  LowerInterTile.cpp
  RewriteDescriptorLayout.cpp
  RewriteDescriptorLayout/Classify.cpp
  RewriteDescriptorLayout/ContractionSynthesis.cpp
  UnaliasLinalgOuts.cpp
  DistributeWork.cpp
  MaterializeBaseAddresses.cpp
  Utility.cpp

  DEPENDS
  KTDPTransformsIncGen

  LINK_LIBS PUBLIC
  KTIR::KtdpDialect
  TritonIR
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
)
