# The tts dialect's declarations.  Two tablegen inputs: the dialect itself, and
# the authoring ops.  No -gen-typedef-decls / -gen-attrdef-decls -- the dialect
# defines no type and no attribute type, and must not, since the point of a
# discardable `tts.` name is that a consumer which does not load THIS dialect can
# still parse it.  Using an attribute another dialect defines is a different
# thing and is allowed: see the note on `kPinAttrName` in TTSDialect.td.
#
# third_party/spyre/CMakeLists.txt already puts the source and binary include/
# trees on the path, which is what every `#include "Dialect/TTS/..."` resolves
# against.  The Triton include tree is on the path from the top-level
# CMakeLists.txt, which is what lets TTSOps.td pull in TritonTypes.td for
# TT_TensorDescType.
#
# The ktir frontend's tree is added HERE and not there, because tablegen is the
# only thing that needs it: `tts.pin` types its memory space as ktdp's own
# `#ktdp.memory_space`, so TTSOps.td includes KTDPAttrs.td.  C++ reaches the same
# headers through the KTIR::KtdpDialect target's interface, which tablegen has no
# equivalent of -- an include path is the only way to tell it where to look.
# KTIR_MLIR_FRONTEND_DIR is set in third_party/spyre/CMakeLists.txt and reaches
# here by ordinary variable scoping.  No DEPENDS is needed for it: what is
# included is a checked-in .td, not a generated one.
include_directories(${KTIR_MLIR_FRONTEND_DIR}/include)
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
