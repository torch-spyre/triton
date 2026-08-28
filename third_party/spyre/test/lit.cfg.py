import os
import shutil

import lit.formats
import lit.llvm

llvm_config = lit.llvm.llvm_config

config.name = "SpyreTriton"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.spyre_obj_root, "test")

llvm_config.with_system_environment([
    "HOME", "INCLUDE", "LIB", "TMP", "TEMP",
    "TRITON_SPYRE_DBO_OPT", "TRITON_SPYRE_DEVICE",
    # Where dbo-opt finds its own default device when TRITON_SPYRE_DEVICE is
    # unset, which resolve_device() treats as a legitimate configuration rather
    # than an error. So this belongs to the ``dbo-opt`` compile test, not to any
    # device launch: drop it and that test compiles against no device on a machine
    # that was relying on the fallback.
    "DEEPTOOLS_PATH",
])

# A ``dbo-opt`` feature, so a test needing the backend compiler says
# ``REQUIRES: dbo-opt`` and lit reports Unsupported when there is none.
#
# That is the whole reason to gate at this level: these tests run pytest, and
# pytest exits 0 when it skips, so a gate living only inside the test shows up as
# an ordinary Passed -- a run that compiled nothing would look exactly like one
# that did.
#
# The real resolver is resolve_dbo_opt() in backend/compiler.py. This is not a
# second opinion but the same rule re-spelled -- a value containing a path
# separator is taken literally, a bare name is looked up on PATH -- because
# importing the backend here would pull triton into lit's config phase. Keep the
# two in step: if they ever disagree, lit runs a file whose tests then skip, and
# the skip is invisible again.
_dbo_opt = os.environ.get("TRITON_SPYRE_DBO_OPT", "dbo-opt")
_dbo_path = _dbo_opt if os.sep in _dbo_opt else shutil.which(_dbo_opt)
if _dbo_path and os.path.isfile(_dbo_path):
    config.available_features.add("dbo-opt")

llvm_config.use_default_substitutions()

config.spyre_tools_dir = os.path.join(config.spyre_obj_root, "bin",
                                      "spyre-triton-opt")
tool_dirs = [config.spyre_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["spyre-triton-opt", "FileCheck"], tool_dirs)

config.substitutions = [
    (key, '"%s"' % val if val and " " in val and not val.startswith('"') else val)
    for key, val in config.substitutions
]
