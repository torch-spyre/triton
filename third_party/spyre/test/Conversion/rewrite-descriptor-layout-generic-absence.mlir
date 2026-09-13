// RUN: spyre-triton-opt %s --split-input-file --rewrite-descriptor-layout-generic | FileCheck %s

// Unannotated IR must come out unchanged. There is no marker, so Phase 1
// physicalizes nothing, nothing is seeded as physical, and the rewrite finds
// every op consistent -- which is what makes an unannotated kernel a no-op
// rather than something the pass has an opinion about.

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @no_marker
// CHECK:         ktdp.construct_memory_view %{{.*}}, sizes: [64, 128], strides: [128, 1]
// CHECK-SAME:      memref<64x128xf32>
// CHECK:         ktdp.construct_access_tile %{{.*}} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
// CHECK:         ktdp.load %{{.*}} : <64x128xindex> -> tensor<64x128xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<64x128xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<64x128xf32>)
// CHECK:         ktdp.store %{{.*}} : tensor<64x128xf32>, <64x128xindex>
tt.func @no_marker(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %i = builtin.unrealized_conversion_cast %in : !tt.ptr<f32> to index
  %iv = ktdp.construct_memory_view %i, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %it = ktdp.construct_access_tile %iv[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %a = ktdp.load %it : <64x128xindex> -> tensor<64x128xf32>
  %o = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %o, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// A bad data-layout option is rejected rather than silently treated as one of
// the two valid values: the pass is invocable directly, bypassing the
// frontend's own validation.

// RUN: not spyre-triton-opt %s --rewrite-descriptor-layout-generic=data-layout=bogus 2>&1 | FileCheck %s --check-prefix=BADOPT
// BADOPT: data-layout must be 'device' or 'host', got 'bogus'
module {
tt.func @opt_validation() {
  tt.return
}
}
