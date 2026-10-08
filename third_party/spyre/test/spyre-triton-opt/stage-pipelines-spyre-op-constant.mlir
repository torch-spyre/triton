// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s --check-prefix=KTIR
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PHYS

// A `tl.spyre_op` whose operand is a constant, the shape
// `tl.spyre_op("sigmoid", tl.full([128], 1.0, tl.float32))` traces to.
//
// What happens, stage by stage. In `ktir` the body is isolated from the
// constant, so nothing folds while the op exists, and the artifact holds the
// fallback, hinted, reading the constant. In `spyrecode` the first greedy
// driver, NormalizeForDevice, folds every hinted op, since each reads only
// constants: the call site becomes one constant, sigmoid(1.0), stored as it is.
// No hinted op is left, so neither FuseComputeAndDataMovement nor
// LowerSpyreOps sees the call site, nothing is selected and nothing is
// reported.

module {
  tt.func public @constant_operand(%out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %n = arith.constant 128 : i32
    %sn = arith.constant 1 : i64
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%n], [%sn] : <f32>, <128xf32>
    %x = arith.constant dense<1.000000e+00> : tensor<128xf32>
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

// KTIR-LABEL:   func.func @constant_operand(
// KTIR:           %[[X:.*]] = arith.constant dense<1.000000e+00> : tensor<128xf32>
// KTIR:           arith.subf %{{.*}}, %[[X]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// KTIR:           math.exp {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// KTIR:           %[[R:.*]] = arith.divf {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// KTIR:           ktdp.store %[[R]]

// PHYS-LABEL:   func.func @constant_operand(
// PHYS:           %[[C:.*]] = arith.constant dense<0.731058597> : tensor<128xf32>
// PHYS-NOT:       linalg.generic
// PHYS-NOT:       spyreop.
// PHYS-NOT:       tts.spyreop_hint
// PHYS:           ktdp.store %[[C]]
