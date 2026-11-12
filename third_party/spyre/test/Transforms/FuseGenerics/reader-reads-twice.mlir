// RUN: spyre-triton-opt %s --fuse-generics > %t 2>&1 || true
// RUN: FileCheck %s < %t

// An op that reads the i1 in two operands is one reader, and is named once.
//
// An op's users are listed once per operand it reads the value in, so
// `arith.select %m, %m, %f` is twice in the users of `%m`. It is one reader of
// the compare: the diagnostic names it once. A -verify-diagnostics test could
// not pin this, since it accepts a note repeated.

// CHECK: error: fuse-generics: operand 0 is an i1 tensor passed from one linalg.generic to another
// CHECK: note: made here, by 'arith.andi'
// CHECK: note: read here, by 'arith.select'
// CHECK-NOT: note: read here
// CHECK: note: LowerSpyreOps lowers an i1 only within a group
func.func @reader_reads_the_i1_twice(%x: tensor<8xi1>, %y: tensor<8xi1>, %z: tensor<8xi1>) -> tensor<8xi1> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xi1>, tensor<8xi1>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: i1, %b: i1, %out: i1):
    %v = arith.andi %a, %b : i1
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xi1>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %z : tensor<8xi1>, tensor<8xi1>) outs(%e : tensor<8xi1>) {
  ^bb0(%m: i1, %f: i1, %out: i1):
    %s = arith.select %m, %m, %f : i1
    linalg.yield %s : i1
  } -> tensor<8xi1>
  return %r : tensor<8xi1>
}
