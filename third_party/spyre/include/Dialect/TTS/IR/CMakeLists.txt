# The tts dialect's declarations.  Two tablegen inputs: the dialect itself, and
# the authoring ops.  No -gen-typedef-decls / -gen-attrdef-decls -- the dialect
# defines no type and no attribute type, and must not, since the point of
# `tts.tensor_layout` being a discardable name over builtin values is that a
# consumer which does not load this dialect can still parse it.
#
# No include_directories: third_party/spyre/CMakeLists.txt already puts the
# source and binary include/ trees on the path, which is what every
# `#include "Dialect/TTS/..."` resolves against.  The Triton include tree is on
# the path from the top-level CMakeLists.txt, which is what lets TTSOps.td pull
# in TritonTypes.td for TT_TensorDescType.
set(LLVM_TARGET_DEFINITIONS TTSDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tts)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tts)

set(LLVM_TARGET_DEFINITIONS TTSOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
# No add_mlir_doc: it resolves its output against MLIR_BINARY_DIR, which no
# CMakeLists under third_party/spyre sets, so a doc target here writes to /docs.
# Nothing in this tree generates dialect docs, so there is nothing to set it for.

# One target for both, so every dependent gets all four generated files from a
# single DEPENDS entry.
add_public_tablegen_target(TTSIncGen)
