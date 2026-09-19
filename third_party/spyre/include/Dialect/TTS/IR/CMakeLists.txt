# The tts dialect's declarations.  Only the dialect itself is tablegen'd: there
# are no ops, no types and no attribute types, so there is no -gen-op-decls /
# -gen-typedef-decls / -gen-attrdef-decls here and no `add_mlir_doc` either --
# there is nothing an op-doc pass would have to say.
#
# No include_directories: third_party/spyre/CMakeLists.txt already puts the
# source and binary include/ trees on the path, which is what every
# `#include "Dialect/TTS/..."` resolves against.
set(LLVM_TARGET_DEFINITIONS TTSDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tts)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tts)
add_public_tablegen_target(TTSIncGen)
