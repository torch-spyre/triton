add_triton_library(KTDPTransforms
  ConvertFunctions.cpp
  LowerDescriptorMemory.cpp
  LowerScalarLoad.cpp
  LowerComputeOps.cpp
  LowerSpyreOps.cpp
  LowerInterTile.cpp
  RewriteDescriptorLayout.cpp
  RewriteDescriptorLayout/Classify.cpp
  RewriteDescriptorLayout/ContractionSynthesis.cpp
  RewriteDescriptorLayout/IndexDomain.cpp
  RewriteDescriptorLayout/PhysicalTypeAnalysis.cpp
  RewriteDescriptorLayout/RequirementAnalysis.cpp
  UnaliasLinalgOuts.cpp
  DropReductionInitFill.cpp
  DistributeWork.cpp
  MaterializeBaseAddresses.cpp

  DEPENDS
  KTDPTransformsIncGen

  LINK_LIBS PUBLIC
  KTDPUtils
  KTIR::KtdpDialect
  KTIR::SpyreOpDialect
  TritonIR
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
)
