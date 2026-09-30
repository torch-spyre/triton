# The -name= value decides the generated symbol names: registerTritonToKTIRPasses()
# for the whole group, and the GEN_PASS_* macro spellings the .cpp files use.
# third_party/spyre/CMakeLists.txt already has the source and binary include/
# trees on the path, so no include_directories is needed here.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name=TritonToKTIR)
add_public_tablegen_target(TritonToKTIRIncGen)
