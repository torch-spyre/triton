// RUN: spyre-triton-opt %s --convert-ttir-functions -split-input-file -verify-diagnostics

// Input shapes --convert-ttir-functions rejects up front.
//
// The pass replaces every !tt.ptr function argument with an index. The only
// thing it knows how to fix up on the way is a builtin.unrealized_conversion_cast
// — the placeholder the memory passes leave behind when they read a base pointer
// through getBasePtrAsIndex (third_party/spyre/lib/Dialect/KTDP/Transforms/Utility.cpp).
// After the retype such a cast reads index and produces index, so it folds away.
//
// Any *other* reader of the argument cannot be fixed up. Retyping the argument
// under it leaves that op holding an operand whose type silently changed, which
// the pass has no way to repair. Each case below is therefore rejected with a
// diagnostic naming the limitation, rather than converted into a module that
// fails verification after the pass has already reported success.
//
// Every input here is valid TTIR on entry — parse it without the pass and it
// round-trips. The rejection is this pass's own precondition, not a general MLIR
// rule.
//
// -split-input-file cuts the file at each dashed marker and treats every chunk
// as an independent module, so one rejection does not mask the next.
// `expected-error @below {{...}}` matches a diagnostic reported on the following
// line; the braced text is a substring match, so it pins the limitation without
// pinning the full wording.

// -----
// Shape A: the pointer itself crosses a block boundary as a successor operand.
//
// The retype only looks at the entry block's argument list, so the ^bb1 block
// argument that receives the pointer is invisible to it. Converting this would
// leave `cf.br` passing an index into a block whose argument is still declared
// !tt.ptr<f32>.
//
// Contrast @multi_block_ptr_arg in convert-functions.mlir, which is accepted:
// there the cast comes *before* the branch, so only the resulting index crosses
// the boundary and the entry-block argument's only user is the cast.
tt.func public @ptr_crosses_block(%p: !tt.ptr<f32>) -> index {
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  cf.br ^bb1(%p : !tt.ptr<f32>)
^bb1(%q: !tt.ptr<f32>):
  %i = builtin.unrealized_conversion_cast %q : !tt.ptr<f32> to index
  tt.return %i : index
}

// -----
// Shape B: a tt.load still reading the pointer directly, in a single-block body.
//
// This is what an ordering mistake looks like — the pass running before
// LowerScalarLoad has consumed the load. Without the precondition check the pass
// reports success and emits `tt.load(%arg0 : index)`, and the failure surfaces
// later as a verifier complaint against tt.load, an op that did nothing wrong.
tt.func public @load_reads_ptr(%p: !tt.ptr<f32>) -> f32 {
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  %v = tt.load %p : !tt.ptr<f32>
  tt.return %v : f32
}

// -----
// Shape C: a tt.addptr chain on the argument.
//
// tt.addptr consumes and produces !tt.ptr, so it is a non-cast user in exactly
// the same way. Kept as a separate case because it is the shape a pointer-
// arithmetic kernel produces, and it is reached through a different pass
// (LowerScalarLoad folds addptr chains) than the bare load above.
tt.func public @addptr_reads_ptr(%p: !tt.ptr<f32>, %off: i32) -> f32 {
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  %q = tt.addptr %p, %off : !tt.ptr<f32>, i32
  %v = tt.load %q : !tt.ptr<f32>
  tt.return %v : f32
}

// -----
// A cast user and a non-cast user on the same argument. Pins that the check
// rejects on the presence of a bad user rather than on the absence of a good
// one: an implementation that accepted the argument as soon as it found one cast
// would pass every case above and still miscompile this one.
tt.func public @mixed_users(%p: !tt.ptr<f32>) -> index {
  %i = builtin.unrealized_conversion_cast %p : !tt.ptr<f32> to index
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  %v = tt.load %p : !tt.ptr<f32>
  tt.return %i : index
}

// -----
// The upper endpoint: an unused !tt.ptr argument has no users at all, so it has
// no bad user either. It must be accepted and retyped to index. A check that
// required at least one cast would wrongly reject this.
// -verify-diagnostics fails the run on any diagnostic without a matching
// annotation, so an over-eager rejection here fails the test.
tt.func public @unused_ptr_arg(%p: !tt.ptr<f32>, %n: i32) -> i32 {
  tt.return %n : i32
}

// -----
// A non-cast user of a *non-pointer* argument is irrelevant — the check is
// scoped to !tt.ptr arguments, because those are the only ones whose type
// changes. Pins that the scoping is real rather than incidental.
tt.func public @non_ptr_arg_any_user(%a: i32) -> i32 {
  %b = arith.addi %a, %a : i32
  tt.return %b : i32
}
