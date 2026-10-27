// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s --check-prefix=KTIR
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PHYS
// A prefix of its own, with nothing but NOT directives, so it scans the whole
// output rather than the span between two positive checks.
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=NOIMM
// The same stage with CanonicalizeGenericOperands and FuseGenerics taken OUT of
// it, spelled as the stage's pass list minus those two passes since a registered
// pipeline cannot have a pass removed from the CLI. It is checked with the same
// prefixes as the full stage, which is the claim: removing them changes nothing
// here.
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | spyre-triton-opt --normalize-for-device --drop-reduction-init-fill --convert-elementwise-to-linalg --linalg-generalize-named-ops --unalias-linalg-outs --lower-spyre-ops | FileCheck %s --check-prefix=PHYS
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | spyre-triton-opt --normalize-for-device --drop-reduction-init-fill --convert-elementwise-to-linalg --linalg-generalize-named-ops --unalias-linalg-outs --lower-spyre-ops | FileCheck %s --check-prefix=NOIMM

// That a division by one leaves the stage as the unary intrinsic, with no float
// immediate anywhere -- and that no other pass in the stage is load-bearing for
// it.
//
// No single pass can make this claim, which is why it is here. LowerSpyreOps'
// divf rule chooses between the unary and the binary intrinsic, so what is under
// test is not that choice -- that is the pass's own test -- but that the stage
// delivers a body in the form the rule can read at all.
//
// WHY THE CONSTANT HAS TO GO. A float immediate reaching a Spyre compute unit is
// not read back as it was written, so a divide by a rounded one is not the divide
// that was written. Nothing downstream refuses it -- dbo-opt has no diagnostic for
// a float immediate operand -- so a regression here is a wrong answer, not a
// failure. `reduce/softmax_on_stick` is the kernel that depends on it; this file is
// the same claim at a size a lit test can read.
//
// THE LAST TWO RUN LINES ARE THE INTERESTING ONES. They assert a
// NON-dependency, and only for this rule: the reciprocal reads its numerator
// through the generic's body, so a splat `ins` and a folded-in scalar constant
// answer the same question, and dropping CanonicalizeGenericOperands leaves the
// result identical. Matching on the body value instead would make that pass
// load-bearing here -- its SplatOperandToScalar is the only thing in the
// pipeline that folds a splat constant into a body -- and the failure would
// surface as a numerical answer rather than as a diff. The two prefixes are
// reused rather than given negated twins precisely so the two spellings cannot
// drift.
//
// The compare rule is the opposite case and is NOT covered here: it genuinely
// needs FuseGenerics, because its group spans two tensor ops and arrives as two
// generics. test/Transforms/LowerSpyreOps/compare.mlir names that pass in its
// own RUN line for exactly that reason.
//
// The kernel is the smallest thing carrying the shape: a 1-D reciprocal whose
// numerator is a splat `arith.constant` beside the divide, which is how
// `tl.full([N], 1.0)` arrives. No descriptor layout and no reduce -- what is under
// test is the sequence, not what any one pass does to a compute op.

module {
  tt.func public @recip_kernel(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %n = arith.constant 1024 : i32
    %s = arith.constant 1 : i64
    %one = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %pid = tt.get_program_id x : i32
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%n], [%s] : <f32>, <1024xf32>
    %o_desc = tt.make_tensor_descriptor %out_ptr, [%n], [%s] : <f32>, <1024xf32>
    %off = arith.muli %pid, %n : i32
    %x = tt.descriptor_load %x_desc[%off] : !tt.tensordesc<1024xf32> -> tensor<1024xf32>
    %r = arith.divf %one, %x : tensor<1024xf32>
    tt.descriptor_store %o_desc[%off], %r : !tt.tensordesc<1024xf32>, tensor<1024xf32>
    tt.return
  }
}

// The `ktir` artifact keeps the divide as written, on tensors and in arith. No
// spyreop op belongs to this stage at all -- both passes that make one are in
// `spyrecode` -- so a kernel that stops here is the arithmetic its author wrote.
//
// KTIR-LABEL: func.func @recip_kernel
// KTIR: arith.constant dense<1.000000e+00> : tensor<1024xf32>
// KTIR: arith.divf
// KTIR-NOT: spyreop

// Through `spyrecode`, the divide is gone and the unary intrinsic is what a
// scalar body holds.
//
// PHYS-LABEL: func.func @recip_kernel
// PHYS: linalg.generic
// PHYS: spyreop.reciprocal

// And the immediate is gone with it, scanned over the whole output. Both spellings
// are named: the splat the author wrote, and the scalar a fold would have left in
// the body.
//
// NOIMM-NOT: spyreop.realdiv
// NOIMM-NOT: arith.constant dense<1.000000e+00>
// NOIMM-NOT: arith.constant 1.000000e+00
