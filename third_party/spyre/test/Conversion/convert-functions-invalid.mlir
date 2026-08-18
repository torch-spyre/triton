// RUN: spyre-triton-opt %s --convert-ttir-functions -split-input-file -verify-diagnostics

// Input shapes --convert-ttir-functions rejects up front.
//
// The pass replaces every !tt.ptr function argument with an index. The only user
// it can fix up is a builtin.unrealized_conversion_cast — the placeholder the
// memory passes leave behind via getBasePtrAsIndex (Utility.cpp). After the
// retype that cast reads index and produces index, so it folds away.
//
// Any other reader is left holding an operand whose type changed underneath it,
// which the pass cannot repair. Those inputs are rejected with a diagnostic
// instead of being converted into a module that fails verification after the
// pass has already reported success.
//
// Every input here is valid TTIR on entry. The rejection is this pass's own
// precondition, not a general MLIR rule.
//
// -split-input-file treats each chunk between dashed markers as its own module,
// so one rejection does not mask the next. `expected-error @below {{...}}`
// matches a diagnostic on the following line, by substring.

// -----
// Shape A: the pointer itself crosses a block boundary as a successor operand.
//
// The retype only looks at the entry block's arguments, so ^bb1's argument is
// invisible to it. Converting this would leave `cf.br` passing an index into a
// block whose argument is still !tt.ptr<f32>.
//
// Contrast @multi_block_ptr_arg in convert-functions.mlir, which is accepted:
// there the cast comes *before* the branch, so only the index crosses the
// boundary.
tt.func public @ptr_crosses_block(%p: !tt.ptr<f32>) -> index {
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  cf.br ^bb1(%p : !tt.ptr<f32>)
^bb1(%q: !tt.ptr<f32>):
  %i = builtin.unrealized_conversion_cast %q : !tt.ptr<f32> to index
  tt.return %i : index
}

// -----
// Shape B: a tt.load still reading the pointer directly.
//
// This is what a pass ordering mistake looks like: this pass ran before
// LowerScalarLoad consumed the load. Without the check, the pass reports success
// and emits `tt.load(%arg0 : index)`, and the failure surfaces later as a
// verifier complaint against tt.load — an op that did nothing wrong.
tt.func public @load_reads_ptr(%p: !tt.ptr<f32>) -> f32 {
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  %v = tt.load %p : !tt.ptr<f32>
  tt.return %v : f32
}

// -----
// Shape C: a tt.addptr chain on the argument.
//
// tt.addptr consumes and produces !tt.ptr, so it is a non-cast user in the same
// way as the load above. Kept separate because it is the shape a pointer-
// arithmetic kernel produces: LowerScalarLoad walks the chain back to its root
// pointer and sweeps the addptr ops only once they are dead, so one left behind
// still reaches this pass.
tt.func public @addptr_reads_ptr(%p: !tt.ptr<f32>, %off: i32) -> f32 {
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  %q = tt.addptr %p, %off : !tt.ptr<f32>, i32
  %v = tt.load %q : !tt.ptr<f32>
  tt.return %v : f32
}

// -----
// A cast user and a non-cast user on the same argument. Pins that the check
// rejects on the presence of a bad user, not the absence of a good one: a check
// that accepted the argument as soon as it found one cast would pass every case
// above and still miscompile this one.
tt.func public @mixed_users(%p: !tt.ptr<f32>) -> index {
  %i = builtin.unrealized_conversion_cast %p : !tt.ptr<f32> to index
  // expected-error @below {{is used by an op that is not an unrealized_conversion_cast}}
  %v = tt.load %p : !tt.ptr<f32>
  tt.return %i : index
}

// -----
// An unused !tt.ptr argument has no users, so it has no bad user either. It must
// be accepted and retyped. A check requiring at least one cast would reject it.
// -verify-diagnostics fails on any unannotated diagnostic, so an over-eager
// rejection here fails the test.
tt.func public @unused_ptr_arg(%p: !tt.ptr<f32>, %n: i32) -> i32 {
  tt.return %n : i32
}

// -----
// A non-cast user of a *non-pointer* argument is fine: the check is scoped to
// !tt.ptr arguments, the only ones whose type changes.
tt.func public @non_ptr_arg_any_user(%a: i32) -> i32 {
  %b = arith.addi %a, %a : i32
  tt.return %b : i32
}
