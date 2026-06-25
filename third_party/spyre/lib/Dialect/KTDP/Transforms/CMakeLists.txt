add_triton_library(KTDPTransforms
  ConvertFunctions.cpp
  LowerDescriptorMemory.cpp
  LowerComputeOps.cpp
  RewriteDescriptorLayout.cpp
  DistributeWork.cpp
  Utility.cpp

  DEPENDS
  KTDPTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRKtdp
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
