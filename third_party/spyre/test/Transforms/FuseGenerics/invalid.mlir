// RUN: spyre-triton-opt %s --fuse-generics -split-input-file -verify-diagnostics

// legalizeAfterFusion, the check step run after the patterns converge: an i1
// tensor passed from one linalg.generic to another is an error at the
// consumer. Its notes name the op that made the i1, each reader, and the group
// shapes LowerSpyreOps lowers an i1 in. Every edge GroupComputes declines leaves
// such an i1, so its negatives are here, in four sections:
//
//   1  never lowerable: the anchor, its predicate or its reader has no device
//      form;
//   2  not lowered yet: a compare read by more than one op;
//   3  declined by a FuseGenerics check, the i1 left between the generics as a
//      consequence: a reduction producer, the shared-producer rule, and the
//      checks on the consumer's operand list.
//
// The in-body counterparts are LowerSpyreOps' invalid.mlir (1) and
// not-yet-lowered.mlir (2).

//===----------------------------------------------------------------------===//
// 1. Never lowerable
//===----------------------------------------------------------------------===//

// The producer yields an arith.andi of two compares, which forms no group.
func.func @andi_chain(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %u = arith.cmpf ogt, %a, %b : f16
    %v = arith.cmpf olt, %a, %b : f16
    // expected-note @below {{made here, by 'arith.andi'}}
    %w = arith.andi %u, %v : i1
    linalg.yield %w : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// A reader that is not selectable: `ord` has no spyreop.compare counterpart.
func.func @ord_uitofp(%x: tensor<8xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf ord, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %m : i1 to f16
    linalg.yield %f : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// Never lowerable: spyreop.select takes one type for its condition and values,
// and spyreop has no float-width conversion.
// A reader that is not selectable: an f32 compare read by an f16 select.
func.func @width_mismatch(%x: tensor<8xf32>, %y: tensor<8xf32>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf32>, tensor<8xf32>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f32, %b: f32, %out: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf ogt, %a, %b : f32
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// Never lowerable: an i1 read as signed is 0 or -1, so `arith.sitofp` gives -1.0
// where the predicate holds, which no spyreop.compare result is.
// A reader that is not selectable: arith.sitofp, where compare-cast takes
// arith.uitofp.
func.func @sitofp(%x: tensor<8xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf oeq, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    // expected-note @below {{read here, by 'arith.sitofp'}}
    %f = arith.sitofp %m : i1 to f16
    linalg.yield %f : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// 2. Not lowered yet: more than one reader
//===----------------------------------------------------------------------===//

// A compare read by both an arith.uitofp and an arith.select: a group takes
// exactly one reader, so the edge is not grouped. Pins the decline; LowerSpyreOps'
// SelectCompareGroup notes sharing one `spyreop.compare` mask between such
// readers as a possible extension.
// Tracked by #254.
func.func @both_readers(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf olt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p : tensor<8xi1>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %out: f16):
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %m : i1 to f16
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %m, %a, %f : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// 3. Declined by a FuseGenerics check
//===----------------------------------------------------------------------===//

// An i1 made by a reduction, here an arith.ori.
func.func @reduction_i1(%x: tensor<8x4xf16>, %y: tensor<8x4xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x, %y : tensor<8x4xf16>, tensor<8x4xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %acc: i1):
    %u = arith.cmpf ogt, %a, %b : f16
    // expected-note @below {{made here, by 'arith.ori'}}
    %w = arith.ori %u, %acc : i1
    linalg.yield %w : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// A reduction producer yielding an arith.cmpf is claimed, and declined
// because the producer has a reduction loop.
func.func @reduction_cmp(%x: tensor<8x4xf16>, %y: tensor<8x4xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>, %e1: tensor<8xi1>) -> tensor<8xf16> {
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x, %y : tensor<8x4xf16>, tensor<8x4xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %acc: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %u = arith.cmpf ogt, %a, %b : f16
    linalg.yield %u : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// A producer whose body is more than the compare, with a second user, is
// not duplicated.
func.func @multi_op_two_readers(%x: tensor<8xf16>, %y: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xi1>) {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %d = arith.subf %a, %b : f16
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf ogt, %d, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %m : i1 to f16
    linalg.yield %f : f16
  } -> tensor<8xf16>
  return %r, %c : tensor<8xf16>, tensor<8xi1>
}

// -----

// A value would be both an ins and an outs of the consumer: the producer's ins
// is the consumer's outs.
func.func @d4_fusion(%x: tensor<8xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf ogt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%x : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %m : i1 to f16
    linalg.yield %f : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}

// -----

// The producer reads an input under a linearizing map, written by hand:
// CanonicalizeGenericOperands leaves a linearizing reshape a tensor op. The
// fused op's map for that input, (d0) -> (d0 floordiv 2, d0 mod 2), has
// results that are neither a bare dim nor a constant.
func.func @linearizing_producer_input(%s: tensor<64x2xf16>, %t: tensor<128xf16>) -> tensor<128xf16> {
  %e1 = tensor.empty() : tensor<128xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%s, %t : tensor<64x2xf16>, tensor<128xf16>) outs(%e1 : tensor<128xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    // expected-note @below {{made here, by 'arith.cmpf'}}
    %v = arith.cmpf ogt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<128xi1>
  %e = tensor.empty() : tensor<128xf16>
  // expected-error @below {{fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another; no spyreop op produces or}}
  // expected-note @below {{LowerSpyreOps lowers an i1 only within a group}}
  %m = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<128xi1>) outs(%e : tensor<128xf16>) {
  ^bb0(%in: i1, %out: f16):
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %in : i1 to f16
    linalg.yield %f : f16
  } -> tensor<128xf16>
  return %m : tensor<128xf16>
}
