#ifndef TTS_OPS
#define TTS_OPS

include "mlir/IR/OpBase.td"
include "Dialect/TTS/IR/TTSDialect.td"
// For TT_TensorDescType. This is the one direction the dependency may point:
// `tts` names a Triton type, Triton names nothing of ours. TritonTypes.td
// defines no op, so `-gen-op-decls` over this file still emits only ours.
include "triton/Dialect/Triton/IR/TritonTypes.td"

class TTS_Op<string mnemonic, list<Trait> traits = []>
    : Op<TTS_Dialect, mnemonic, traits>;

def TTS_TensorLayoutOp : TTS_Op<"tensor_layout", []> {
  let summary = "Annotate a tensor descriptor with its physical device layout";

  let description = [{
    `tts.tensor_layout` is the **authoring** spelling of the physical
    (stick-tiled) device layout for a `tt.make_tensor_descriptor`: three
    parallel i64 array attributes (`phys_src` / `phys_op` / `phys_arg`), one
    entry per physical dim, in the OpSpec `device_coordinates` form. See the
    `tts` dialect's `kTensorLayoutAttrName` note for what the three arrays mean.

    No result; the op is a marker. `LowerTTSMarkers` consumes it: the three
    arrays are written onto the `ktdp.construct_memory_view` built for the
    descriptor as the `tts.tensor_layout` **attribute**, and the op is erased
    along with the `unrealized_conversion_cast` that bridged the view's memref
    back to `!tt.tensordesc` for this operand's sake. The view and that cast are
    `LowerDescriptorMemory`'s, which is why this marker cannot be lowered before
    that pass has run.

    So the dialect spells the same layout twice, and the two spellings are a
    before/after pair rather than alternatives:

      op        — what a kernel author writes, on a `!tt.tensordesc`
      attribute — what the lowered IR carries, on the memory view

    The structural rules on the three arrays are the attribute's rules, checked
    by the same `tts::verifyTensorLayoutArrays`. The only difference is where the
    logical rank comes from: here it is the descriptor's block type, there it is
    the memory view's memref. Same rank, different source.
  }];

  let arguments = (ins
    TT_TensorDescType:$desc,
    DenseI64ArrayAttr:$phys_src,
    DenseI64ArrayAttr:$phys_op,
    DenseI64ArrayAttr:$phys_arg
  );

  let assemblyFormat = "$desc attr-dict `:` type($desc)";
  let hasVerifier = 1;
}

def TTS_PinOp : TTS_Op<"pin", []> {
  let summary = "Place a named value's buffer in a chosen memory space";

  let description = [{
    `tts.pin` is the **authoring** spelling of where an intermediate's buffer
    lives. The scheduler admits one compute per local schedule, so a value handed
    from one compute to the next has to go through memory; this op is how a
    kernel says *which* memory, and for the scratchpad, where in it.

    No result; the op is a marker, and the value it names keeps its own type and
    its own uses. `LowerTTSMarkers` consumes it, writing the `tts.pin`
    **attribute** onto the op PRODUCING the pinned value -- so like
    `tts.tensor_layout` this is the authoring half of a before/after pair, and
    what differs is only where the attribute lands.

    The pin names a **value** and not a `!tt.tensordesc`: a descriptor takes its
    element type from the `!tt.ptr` it is based on, so giving one a scratchpad
    location would mean supplying a dtype as well, and a value has a type already.

    `memory_space` is a `ktdp::MemorySpaceKind` NAME held as a string, and this is
    the one field whose spelling is forced rather than chosen. Building
    `#ktdp.memory_space` needs ktdp LOADED; ktdp declares `mlir::func::FuncDialect`
    dependent (KTDPDialect.td), and loading `func` makes it promise a
    `DialectInlinerInterface` nothing in this tree registers, so the `ttir` stage's
    Inliner aborts the compile. This op is built during TRACING, so the one place
    that attribute would be constructed is the one place ktdp must not be loaded.
    `MaterializePinnedBuffers` symbolizes the name in `spyrecode`, where the dialect
    is loaded anyway. A string also cannot carry a `ct_id`, which is the right shape:
    a pin means whichever core runs the kernel, so `ct_id = 7` is not expressible
    rather than refused.

    Only `ct_local` is admitted -- an intermediate in HBM is written as a
    `tl.make_tensor_descriptor` with an explicit store and load, and nothing here
    allocates an anonymous global buffer -- and the field stays a memory space so the
    surface does not change shape if `global` ever becomes placeable.

    `offset` is an **element index**, the unit `ktdp.construct_memory_view`'s
    `$offset` operand feeds, counted from the base of whatever allocation the
    scratchpad allocator gives this kernel and NOT from the start of the scratchpad.
    That decides which rules exist: alignment is the allocator's, since it aligns the
    base it hands out, and what survives is what holds for any base -- offset plus
    extent within one scratchpad, and no two pins of one kernel overlapping. Both are
    `MaterializePinnedBuffers`'.

    Optional in ODS and REQUIRED by the verifier, which is the honest statement of
    what exists: an offsetless pin is the design's baseline, since the compiler places
    every intermediate and a pin only overrides where, but nothing here can act on one
    -- no placement analysis, and nothing allocates a buffer that is not a kernel
    argument. Optional is what lets that refusal lift without the surface changing.

    An ATTRIBUTE and not an operand, which is what lets the marker become an
    attribute on the producing op -- an operand would have nowhere to go -- and a
    single `i32`, so the region a pin occupies is known here by arithmetic. ONE
    offset for every core, with no per-core spelling, because the scratchpad is
    per-core: an offset varying with the program id would move each core's own value
    within its own scratchpad rather than spread a tensor across cores, and every
    core would then reserve the hull of the set. A per-core form stays additive if
    something ever wants it. An affine offset could not come back either way, since
    `tl.program_id(0)` is a run-time read until DistributeWork folds it against the
    grid, so nothing at trace time can evaluate `base + pid*stride` into an attribute.

    Example:
    ```mlir
      tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<11x2049xf16>
    ```
  }];

  let arguments = (ins
    AnyStaticShapeTensor:$value,
    StrAttr:$memory_space,
    OptionalAttr<I32Attr>:$offset
  );

  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

// Deliberately NOT `Pure`, unlike the ktdp view ops it lowers to. Those
// materialize addressing and nothing else; this one's call is an implicit barrier
// over the participant set, so an op with no side-effect information -- which is
// conservatively treated as having some -- is the honest marking. It also keeps
// the op alive when its result is unused, which is a kernel to diagnose rather
// than to silently drop.
def TTS_MakeDistributedDescriptorOp
    : TTS_Op<"make_distributed_descriptor", []> {
  let summary = "Compose every core's share of a tensor into one descriptor";

  let description = [{
    `tts.make_distributed_descriptor` is the **authoring** spelling of a
    redistribution. Each instance passes ITS OWN share and gets back a descriptor
    over the whole tensor; reading it at an offset of the instance's choosing is
    the redistribution, and the offsets differing per instance is what makes it
    one.

    This op has a RESULT, which makes it the odd one out in this dialect -- the
    other two name a value in order to say something about it and are erased. This
    one is a value's authoring form, and the value is a `!tt.tensordesc` so that
    `.load()` on it is the ordinary descriptor read and needs no new surface.

    It is a **collective constructor**: `tt.make_tensor_descriptor` takes a
    pointer, the same in every instance, while this takes a TENSOR, a different
    value in every instance, so its result depends on all of them. The authoring
    vantage stays per-instance all the same -- no instance names a `ct_id`, none
    enumerates cores, and each writes only its own offsets. Two consequences the
    design doc states rather than leaves to be discovered: the call is an implicit
    barrier over the participant set, and `.load()` on the result is a TRANSFER
    that is syntactically identical to a local read.

    The attributes, and note there is no destination table -- the destination
    arrangement is the offset each instance passes to `.load()`:

      work_slices  the PARTITION table, one entry per region the view is composed
                   from, NOT one per tile. Each entry is a dictionary of
                   dimension key to slice index, all entries carrying the same
                   keys, spelled as `tl.inter_tile`'s `work_slices` is. The slice
                   count along a key is one more than the largest index appearing
                   under it, and is deliberately not written down.
      axes         one entry per tensor dimension, naming the partition key that
                   dimension is divided along, or `""` for a dimension the work
                   was not divided on.
      block_shape  the extent of one `.load()`, which is also the result
                   descriptor's block shape. It has NO relation to the share's
                   type and is bounded by the COMPOSED extent instead, which is
                   the share's times the slice count. Less than a share is a
                   relayout reading the region this instance ends up holding;
                   exactly a share is one region per load; more is a gather
                   spanning several partitions, and at the limit the whole
                   composed axis, which is what makes a fold over it an
                   all-reduce.

    Ownership is a table rather than an axis count because the measured patterns
    are frequently strided -- cores two apart, or eight apart, or only the
    even-numbered ones -- and no count reproduces those. `axes` is not that form
    either: it maps a dimension to a key and says nothing about counts or holders.

    Where the share LIVES is not stated here. It is stated by the `tts.pin` on the
    same value, and the lowering recovers it from the buffer that pin was placed
    in. So a kernel that composes must pin, and that is the whole reason `tts.pin`
    exists.

    Example:
    ```mlir
      %whole = tts.make_distributed_descriptor %share
          {work_slices = [ {n = 0 : i64}, {n = 1 : i64} ],
           axes = ["", "n"], block_shape = array<i64: 64, 64>}
          : tensor<64x64xf16> -> !tt.tensordesc<64x64xf16>

    (The spaces inside `work_slices` are this comment's, not the printer's: a
    closing brace immediately followed by a closing bracket would end the
    TableGen code block this description lives in.)
    ```
  }];

  let arguments = (ins
    AnyStaticShapeTensor:$partial,
    ArrayAttr:$work_slices,
    ArrayAttr:$axes,
    DenseI64ArrayAttr:$block_shape
  );

  let results = (outs TT_TensorDescType:$result);

  let assemblyFormat = "$partial attr-dict `:` type($partial) `->` type($result)";
  let hasVerifier = 1;
}

#endif // TTS_OPS
