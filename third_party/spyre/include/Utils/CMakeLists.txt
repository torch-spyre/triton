# Header-only: Utility.h declares the helpers the SpyreUtils library defines.
# No tablegen, and no include_directories -- third_party/spyre/CMakeLists.txt
# already puts the source and binary include/ trees on the path.
#
# A bare `Utils/` at the include root sits beside MLIR's `mlir/` and Triton's
# `triton/` because spyre's include root carries no project-name prefix.  That
# is safe rather than lucky: the root is added by a directory-scoped
# include_directories in third_party/spyre/CMakeLists.txt, so a sibling
# backend's identically named header (third_party/amd/include/Utils/Utility.h)
# is never on the path for a translation unit compiled here, in a co-build or
# otherwise.
