// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// Op-verifier tests for tt.spyre_tensor_layout. The three coord arrays are
// parallel (one entry per physical dim); the verifier rejects shapes the
// RewriteDescriptorLayout consumers cannot interpret, which previously
// crashed the pass instead of diagnosing.

// phys_op has 2 entries but phys_src/phys_arg have 3
tt.func @size_mismatch(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{phys_src, phys_op and phys_arg must have the same number of entries}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// no physical dims at all
tt.func @empty_coords(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{must describe at least one physical dim}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64>,
     phys_op = array<i64>,
     phys_arg = array<i64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// phys_op[0] = 3 is not a valid CoordOp (0=identity, 1=floordiv, 2=mod)
tt.func @bad_op_code(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{phys_op[0] must be 0 (identity), 1 (floordiv) or 2 (mod), got 3}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 3, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// negative phys_src
tt.func @negative_src(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{phys_src[1] must be in [0, 2), got -1}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, -1, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// phys_src[0] = 2 exceeds the rank-2 descriptor's logical rank
tt.func @src_out_of_range(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{phys_src[0] must be in [0, 2), got 2}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 2, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// logical dim 0 claimed twice, both identity — not a floordiv+mod stick split
tt.func @duplicate_identity(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{logical dim 0 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry), got 2 identity, 0 floordiv, 0 mod}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 0, 0, 1>,
     phys_op = array<i64: 0, 0, 0>,
     phys_arg = array<i64: 0, 0, 0>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// logical dim 1 split floordiv+floordiv instead of floordiv+mod
tt.func @duplicate_floordiv(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{logical dim 1 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry), got 0 identity, 2 floordiv, 0 mod}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 1>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// logical dim 1 split three ways
tt.func @triple_split(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{logical dim 1 appears in 3 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry), got 1 identity, 1 floordiv, 1 mod}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, 1, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// floordiv divisor of 0 would divide by zero when deriving physical extents
tt.func @zero_floordiv_arg(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{phys_arg[0] must be > 0 for a floordiv/mod dim, got 0}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 0, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// mod modulus of 0 would make the stick width zero
tt.func @zero_mod_arg(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{phys_arg[2] must be > 0 for a floordiv/mod dim, got 0}}
  tt.spyre_tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 0>} : !tt.tensordesc<64x64xf32>
  tt.return
}
