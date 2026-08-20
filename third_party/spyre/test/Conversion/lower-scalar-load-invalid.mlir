// RUN: spyre-triton-opt %s --lower-scalar-load -split-input-file -verify-diagnostics

// Negative tests for --lower-scalar-load: the masked-load precondition. A
// masked scalar `tt.load`'s mask must be a compile-time constant (a
// materialized `arith.constant` i1) — Spyre has no runtime control-flow
// divergence, so a mask that might depend on runtime data is refused with a
// diagnostic rather than lowered to a runtime branch. `-split-input-file` is
// required here: the precondition check is a `module.walk` that interrupts
// on the first diagnostic, so two cases in one module would emit only one.
//
// Written by hand — utils/generate-test-checks.py has no generator for
// expected-error annotations.

// -----

// The mask is a function argument, not a materialized constant. Mirrors the
// descriptor path's `test_descriptor_from_arg_fails`.

module {
  tt.func @runtime_mask(%ptr: !tt.ptr<f16>, %mask: i1, %other: f16, %out: !tt.ptr<f16>) {
    // expected-error @below {{mask must be a compile-time constant}}
    %v = tt.load %ptr, %mask, %other : !tt.ptr<f16>
    tt.store %out, %v : !tt.ptr<f16>
    tt.return
  }
}

// -----

// A comparison of two constants is *not* folded by `getConstantMask` — only
// a literal `arith.constant` counts. This pins the "we don't fold
// comparisons, only literal constants" behavior, distinct from the
// function-argument case above.

module {
  tt.func @cmpi_mask(%ptr: !tt.ptr<f16>, %other: f16, %out: !tt.ptr<f16>) {
    %a = arith.constant 3 : i32
    %b = arith.constant 5 : i32
    %mask = arith.cmpi slt, %a, %b : i32
    // expected-error @below {{mask must be a compile-time constant}}
    %v = tt.load %ptr, %mask, %other : !tt.ptr<f16>
    tt.store %out, %v : !tt.ptr<f16>
    tt.return
  }
}
