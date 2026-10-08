// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir | FileCheck %s --check-prefix=KTIR
// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PHYS

// A `tl.spyre_op` request across both stages, driven by the two stage flags.
//
// The claim no single pass makes: the request is a tts.spyre_op holding its
// fallback after the `ttir` stage; the `ktir` artifact holds the fallback alone,
// inlined and tagged, in dialects any KTIR reader loads; and in `spyrecode` the
// tagged ops -- one generic each after ConvertElementwiseToLinalg -- are fused
// into one body, physicalized with it, and replaced by the one intrinsic.
//
// Two kernels. The first is one f16 gelu request on a stick-tiled 2-D kernel,
// which is the layout claim: the fused body is physicalized whole. The second is
// two requests back to back, an f32 sigmoid feeding an f32 silu, which is the
// hard case for the fusion clause: it must fuse within a request and not across
// the two.

module {
  tt.func public @gelu(%in_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f16>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %m = arith.constant 64 : i32
    %n = arith.constant 128 : i32
    %sm = arith.constant 128 : i64
    %sn = arith.constant 1 : i64
    %in_desc = tt.make_tensor_descriptor %in_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %in_desc {phys_src = array<i64: 1, 0, 1>,
                                phys_op = array<i64: 1, 0, 2>,
                                phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>,
                                 phys_op = array<i64: 1, 0, 2>,
                                 phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %x = tt.descriptor_load %in_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
    %g = tts.spyre_op "gelu" (%x) : (tensor<64x128xf16>) -> tensor<64x128xf16> {
    ^bb0(%a: tensor<64x128xf16>):
      %half = arith.constant dense<5.000000e-01> : tensor<64x128xf32>
      %one = arith.constant dense<1.000000e+00> : tensor<64x128xf32>
      %rs2 = arith.constant dense<0.707106769> : tensor<64x128xf32>
      %xf = arith.extf %a : tensor<64x128xf16> to tensor<64x128xf32>
      %h = arith.mulf %xf, %half : tensor<64x128xf32>
      %sc = arith.mulf %xf, %rs2 : tensor<64x128xf32>
      %er = math.erf %sc : tensor<64x128xf32>
      %p = arith.addf %er, %one : tensor<64x128xf32>
      %y = arith.mulf %h, %p : tensor<64x128xf32>
      %t = arith.truncf %y : tensor<64x128xf32> to tensor<64x128xf16>
      tts.spyreop_yield %t : tensor<64x128xf16>
    }
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %g : !tt.tensordesc<64x128xf16>, tensor<64x128xf16>
    tt.return
  }
}

// The `ktir` stage. No tts op is left -- an op from a dialect a consumer does
// not load fails at parse -- and the fallback is plain arith/math on tensors,
// every op tagged with its request.
//
// KTIR-LABEL:   func.func @gelu(
// KTIR:           %[[X:.*]] = ktdp.load
// KTIR:           arith.extf %[[X]] {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}}
// KTIR:           math.erf {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}}
// KTIR:           %[[G:.*]] = arith.truncf {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}}
// KTIR:           ktdp.store %[[G]]
// KTIR-NOT:       tts.spyre_op
// KTIR-NOT:       tts.spyreop_yield

// The `spyrecode` stage. The request is ONE body, on the physical 2x64x64
// layout at both ends, holding only its intrinsic: the fallback is gone, casts
// included, and no tag reaches dbo-opt.
//
// PHYS-LABEL:   func.func @gelu(
// PHYS:           %[[X:.*]] = ktdp.load {{.*}} -> tensor<2x64x64xf16>
// PHYS:           %[[G:.*]] = linalg.generic {{.*}} ins(%[[X]] : tensor<2x64x64xf16>) outs(%{{.*}} : tensor<2x64x64xf16>)
// PHYS-NEXT:      ^bb0(%[[A:.*]]: f16, %{{.*}}: f16):
// PHYS-NEXT:        %[[R:.*]] = spyreop.gelu %[[A]] : f16
// PHYS-NEXT:        linalg.yield %[[R]] : f16
// PHYS:           ktdp.store %[[G]]
// PHYS-NOT:       tts.spyreop_hint
// PHYS-NOT:       math.

// -----

module {
  tt.func public @sigmoid_silu(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %n = arith.constant 128 : i32
    %sn = arith.constant 1 : i64
    %in_desc = tt.make_tensor_descriptor %in_ptr, [%n], [%sn] : <f32>, <128xf32>
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%n], [%sn] : <f32>, <128xf32>
    %x = tt.descriptor_load %in_desc[%c0_i32] : !tt.tensordesc<128xf32> -> tensor<128xf32>
    %s = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
    ^bb0(%a: tensor<128xf32>):
      %one = arith.constant dense<1.000000e+00> : tensor<128xf32>
      %zero = arith.constant dense<0.000000e+00> : tensor<128xf32>
      %ng = arith.subf %zero, %a : tensor<128xf32>
      %e = math.exp %ng : tensor<128xf32>
      %d = arith.addf %e, %one : tensor<128xf32>
      %r = arith.divf %one, %d : tensor<128xf32>
      tts.spyreop_yield %r : tensor<128xf32>
    }
    %y = tts.spyre_op "silu" (%s) : (tensor<128xf32>) -> tensor<128xf32> {
    ^bb0(%a: tensor<128xf32>):
      %one = arith.constant dense<1.000000e+00> : tensor<128xf32>
      %zero = arith.constant dense<0.000000e+00> : tensor<128xf32>
      %ng = arith.subf %zero, %a : tensor<128xf32>
      %e = math.exp %ng : tensor<128xf32>
      %d = arith.addf %e, %one : tensor<128xf32>
      %r = arith.divf %a, %d : tensor<128xf32>
      tts.spyreop_yield %r : tensor<128xf32>
    }
    tt.descriptor_store %out_desc[%c0_i32], %y : !tt.tensordesc<128xf32>, tensor<128xf32>
    tt.return
  }
}

// Two call sites, two ids -- even though the canonicalizer is free to merge
// their equal constants, which is why a constant carries no hint.
//
// KTIR-LABEL:   func.func @sigmoid_silu(
// KTIR:           math.exp {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// KTIR:           math.exp {{.*}} {tts.spyreop_hint = {id = 1 : i64, name = "silu"}}

// Two bodies, one intrinsic each, the second reading the first.
//
// PHYS-LABEL:   func.func @sigmoid_silu(
// PHYS:           %[[S:.*]] = linalg.generic
// PHYS-NEXT:      ^bb0(%[[A:.*]]: f32, %{{.*}}: f32):
// PHYS-NEXT:        %[[R:.*]] = spyreop.sigmoid %[[A]] : f32
// PHYS-NEXT:        linalg.yield %[[R]] : f32
// PHYS:           %[[Y:.*]] = linalg.generic {{.*}} ins(%[[S]] : tensor<128xf32>)
// PHYS-NEXT:      ^bb0(%[[B:.*]]: f32, %{{.*}}: f32):
// PHYS-NEXT:        %[[Q:.*]] = spyreop.silu %[[B]] : f32
// PHYS-NEXT:        linalg.yield %[[Q]] : f32
// PHYS:           ktdp.store %[[Y]]
// PHYS-NOT:       tts.spyreop_hint
// PHYS-NOT:       math.
