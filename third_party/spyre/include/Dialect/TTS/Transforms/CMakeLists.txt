# No include_directories here.  third_party/spyre/CMakeLists.txt already puts
# both the source and the binary `include/` trees on the path, which is what
# every `#include "Dialect/TTS/..."` in the tree resolves against.
#
# The -name= value decides the generated symbol names:
# registerTTSTransformsPasses() for the whole group, and the GEN_PASS_* macro
# spellings the .cpp files use.  "TTSTransforms" rather than bare "TTS" so the
# group's symbols cannot collide with anything the dialect itself generates.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name=TTSTransforms)
add_public_tablegen_target(TTSTransformsIncGen)
