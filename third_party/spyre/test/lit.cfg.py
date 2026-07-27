import os
import lit.formats
import lit.llvm

llvm_config = lit.llvm.llvm_config

config.name = "SpyreTriton"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.spyre_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

config.spyre_tools_dir = os.path.join(config.spyre_obj_root, "bin",
                                      "spyre-triton-opt")
tool_dirs = [config.spyre_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["spyre-triton-opt", "FileCheck"], tool_dirs)

config.substitutions = [
    (key, '"%s"' % val if val and " " in val and not val.startswith('"') else val)
    for key, val in config.substitutions
]
