# Transforms whose subject is the KTDP dialect's own abstractions -- memory
# views and access tiles.  RewriteDescriptorLayout/ stays whole inside this
# source directory on purpose: add_triton_library gives each library its own
# source dir as a private include root, which is what makes
# `#include "RewriteDescriptorLayout/Classify.h"` resolve from the twelve places
# that spell it.  Splitting the subdirectory out breaks all of them.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# No func, math, memref or spyreop: this pass touches none of them.
add_triton_library(KTDPTransforms
  RewriteDescriptorLayout.cpp
  RewriteDescriptorLayoutGeneric.cpp
  RewriteDescriptorLayout/Classify.cpp
  RewriteDescriptorLayout/ContractionSynthesis.cpp
  RewriteDescriptorLayout/IndexDomain.cpp
  RewriteDescriptorLayout/PhysicalTypeAnalysis.cpp
  RewriteDescriptorLayout/RequirementAnalysis.cpp

  DEPENDS
  KTDPTransformsIncGen

  LINK_LIBS PUBLIC
  KTDPUtils
  SpyreUtils
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
