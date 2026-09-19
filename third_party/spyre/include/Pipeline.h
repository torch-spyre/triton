//===- Pipeline.h - The Spyre backend's compile stages as pipelines -------===//
//
// One spelling of each pass sequence the backend runs. It used to have two --
// a fused C++ helper and a pass-by-pass loop in backend/compiler.py -- and they
// drifted, with the C++ one installing a different list and nothing in the
// suite able to notice, because every .mlir test drives a single pass.
//
// Two builders, one per compile stage: `ktir` (TTIR in, KTIR out) and
// `spyrecode` (KTIR in, KTIR out, then dbo-opt). The stage boundary is here
// rather than in Python, so moving a pass across it is one edit to one list.
//
// Options are plain values, not tablegen'd or llvm::cl members: both callers
// have them as typed values already -- the Python stage through pybind, the CLI
// through the registration in RegisterEverything.cpp, which is where a string
// becomes one of these.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_PIPELINE_H
#define TRITON_SPYRE_PIPELINE_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace mlir::triton::spyre {

/// Called by buildTTIRToKTIRPipeline once after each pass that a caller can
/// anchor an extra pass on, naming that pass.
///
/// TEMPORARY. It exists only for `SpyreOptions.required_fixes`, which names a
/// pass and an anchor as strings and so has to resolve the pass name on the
/// Python side. The hook appends to the same pass manager, in place, which is
/// what keeps the fix in the position the anchor asks for. It goes when that
/// option does: a fixed ordered list has nothing to anchor to.
using PipelineAnchorHook = std::function<void(llvm::StringRef anchorName)>;

struct TTIRToKTIRPipelineOptions {
  /// RewriteDescriptorLayout's HBM layout: "device" (stickified row-major
  /// physical strides) or "host" (strides derived from the logical ones).
  /// Anything else is read as "host" by the pass itself.
  std::string dataLayout = "device";

  /// DistributeWork's per-axis partition of the hardware grid. Empty leaves
  /// the pass's own default.
  std::vector<int64_t> grid;

  /// Unset for a plain compile; see PipelineAnchorHook.
  PipelineAnchorHook anchorHook;
};

struct SpyrecodePipelineOptions {
  /// Whether to replace the entry function's base-address arguments with
  /// `arith.constant` and drop them from the signature. False leaves them
  /// symbolic, for a runtime that patches the real addresses in at launch.
  ///
  /// A flag of its own rather than "baseAddresses is non-empty", because an
  /// empty list is a legitimate value in the binding mode: it is what a kernel
  /// with no pointer arguments has.
  bool bindBaseAddresses = false;

  /// The addresses to bind, as element indices, positionally. Read only when
  /// bindBaseAddresses.
  std::vector<int64_t> baseAddresses;
};

/// Builds the `ktir` stage: Triton IR in, KTIR out.
///
/// Ends with DistributeWork and a canonicalize, which is the whole stage and
/// not just the dialect conversion -- the split that let the old fused helper
/// drift was exactly that those two were added by the caller.
///
/// Deliberately no CSE: it is unsafe on an author-written HBM round trip, where
/// a store's `ktdp.construct_access_tile` and the matching load's are both Pure
/// and address the same block, so CSE merges them and dbo-opt's compute-group
/// extraction then aborts. See issue #161.
void buildTTIRToKTIRPipeline(OpPassManager &pm,
                             const TTIRToKTIRPipelineOptions &options);

/// Builds the IR half of the `spyrecode` stage: KTIR in, KTIR out, on its way
/// to dbo-opt. The tool invocation is not part of it -- that is the caller's,
/// and keeping it out is what makes the module dbo-opt receives nameable.
///
/// A pass belongs here rather than in the `ktir` stage when either half of that
/// stage's contract fails for it: it is required by dbo-opt rather than by the
/// IR (the `ktir` stage runs for every compile, most of which stop there, and a
/// kernel that never becomes a binary can be one the pass rejects), or its
/// output is no longer standalone KTIR.
void buildSpyrecodePipeline(OpPassManager &pm,
                            const SpyrecodePipelineOptions &options);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_PIPELINE_H
