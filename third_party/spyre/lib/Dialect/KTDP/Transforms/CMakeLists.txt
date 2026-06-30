add_triton_library(KTDPTransforms
  ConvertFunctions.cpp
  LowerDescriptorMemory.cpp
  LowerComputeOps.cpp
  DistributeWork.cpp
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
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
)
