# No include_directories here.  third_party/spyre/CMakeLists.txt already puts
# both the source and the binary `include/` trees on the path, which is what
# every `#include "Dialect/KTDP/..."` in the tree resolves against.
#
# What used to sit here were two relative walks, and both were inert: they
# resolved one level too high (to third_party/spyre rather than
# third_party/spyre/include), and include_directories is directory-scoped, so
# set in an include/ directory they never reached a library target's compile
# line at all.  Confirmed by diffing the -I list in
# build/*/compile_commands.json across their removal.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name=KTDP)
add_public_tablegen_target(KTDPTransformsIncGen)
