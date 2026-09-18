# The -name= value decides the generated symbol names:
# registerSpyreTransformsPasses() for the whole group, and the GEN_PASS_* macro
# spellings the .cpp files use. "Spyre" rather than bare "Transforms" so the
# symbol cannot collide with upstream MLIR's own registerTransformsPasses().
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name=SpyreTransforms)
add_public_tablegen_target(SpyreTransformsIncGen)
