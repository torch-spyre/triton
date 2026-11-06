# The spyreop lowering rules: the group shapes, and the questions they share
# with LowerSpyreOps' one-op patterns. A library of its own because two passes
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
  # spyreop.compare and spyreop.select, which the group shapes build.
  KTIR::SpyreOpDialect
  MLIRIR
  # arith.cmpf, arith.uitofp and arith.select, which the group shapes match.
  MLIRArithDialect
  # linalg.generic, the body a group's anchor is read through.
  MLIRLinalgDialect
)
