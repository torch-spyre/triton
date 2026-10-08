// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir -verify-diagnostics

// A `tl.spyre_op` whose operand is a constant, the shape
// `tl.spyre_op("sigmoid", tl.full([128], 1.0, tl.float32))` traces to once
// canonicalized. Every op of the body would read only constants, so the call
// site would fold to one constant in `spyrecode` and no intrinsic would be
// selected; LowerTTSMarkers refuses it at the end of the `ktir` stage.

module {
  tt.func public @constant_operand(%out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %n = arith.constant 128 : i32
    %sn = arith.constant 1 : i64
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%n], [%sn] : <f32>, <128xf32>
    %x = arith.constant dense<1.000000e+00> : tensor<128xf32>
    // expected-error @+1 {{tl.spyre_op("sigmoid"): operand 0 is a constant, so the call site would fold to a constant and no intrinsic would be selected}}
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
    tt.descriptor_store %out_desc[%c0_i32], %s : !tt.tensordesc<128xf32>, tensor<128xf32>
    tt.return
  }
}
