# The spyreop lowering rules, kSpyreOpLoweringRules, and the questions they
# share with LowerSpyreOps' diagnostics. A library of its own because two passes
# read it -- LowerSpyreOps lowers by it, and FuseGenerics fuses only what it
# would lower -- so neither pass includes the other's code. Sits below
# SpyreTransforms, and links no pass library.
#
# Not in SpyreUtils, which links no dialect of ours: the rules build spyreop
# ops, so KTIR::SpyreOpDialect is a real dependency here.
#
# LINK_LIBS derived from what this file (and its header) include.
add_triton_library(SpyreOpUtils
  SpyreOpLoweringRules.cpp

  LINK_LIBS PUBLIC
  # The spyreop ops the rules build.
  KTIR::SpyreOpDialect
  MLIRIR
  # The arith and math ops the rules match.
  MLIRArithDialect
  MLIRMathDialect
  # linalg.generic, the body a rule's anchor is read through.
  MLIRLinalgDialect
)
