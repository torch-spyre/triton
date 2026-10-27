// RUN: spyre-triton-opt %s --canonicalize-generic-operands --fuse-generics --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// After CanonicalizeGenericOperands and FuseGenerics,
// RewriteDescriptorLayoutGeneric rewrites every annotated descriptor and
// refuses nothing. Both inputs
// are the pass input of a device fixture, verbatim.

// An annotated statistic chain, the `reduce__stat_chain_off_stick` shape. The
// broadcast generic in front of the subtract is absorbed into its operand map,
// so the layout pass reads the statistic at `(d0, d1, d2) -> (d0, d2)`.
//
// CHECK-LABEL: func.func @stat_chain_off_stick
// CHECK:         linalg.generic {{.*}}iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:         %[[STAT:.*]] = ktdp.load {{.*}} -> tensor<2x64xf16>
// CHECK:         linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %[[STAT]] : tensor<2x64x64xf16>, tensor<2x64xf16>)
// CHECK-NEXT:    ^bb0
// CHECK-NEXT:      arith.subf
// CHECK-NOT:     linalg.generic
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
module {
  func.func @stat_chain_off_stick(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [128], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
    %3 = ktdp.construct_access_tile %0[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf16>
    %5 = tensor.empty() : tensor<128xf16>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%4 : tensor<64x128xf16>) outs(%5 : tensor<128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %17 = arith.addf %in, %out : f16
      linalg.yield %17 : f16
    } -> tensor<128xf16>
    %7 = ktdp.construct_access_tile %1[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    ktdp.store %6, %7 : tensor<128xf16>, <128xindex>
    %8 = ktdp.construct_access_tile %0[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
    %9 = ktdp.load %8 : <64x128xindex> -> tensor<64x128xf16>
    %10 = ktdp.construct_access_tile %1[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    %11 = ktdp.load %10 : <128xindex> -> tensor<128xf16>
    %12 = tensor.empty() : tensor<64x128xf16>
    %13 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<128xf16>) outs(%12 : tensor<64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<64x128xf16>
    %14 = tensor.empty() : tensor<64x128xf16>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %13 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%14 : tensor<64x128xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %17 = arith.subf %in, %in_0 : f16
      linalg.yield %17 : f16
    } -> tensor<64x128xf16>
    %16 = ktdp.construct_access_tile %2[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
    ktdp.store %15, %16 : tensor<64x128xf16>, <64x128xindex>
    return
  }
}

// -----

// A `tl.where(mask != zero, ...)` whose zero is a `tl.full` splat, the
// `where__select` shape. FoldSplatConstant drops the splat for a scalar the body
// reads, which the layout pass accepts; left as an `ins` with no layout, the
// layout pass refuses it as needing a linearizing map.
//
// CHECK-LABEL: func.func @where_1d_device
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:         linalg.generic
// CHECK:           arith.cmpf une, %{{.*}}, %{{.*}} : f16
// CHECK:         linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x64xf16>, tensor<2x64xf16>, tensor<2x64xf16>)
// CHECK:           %[[C:.*]] = arith.cmpf une, %{{.*}}, %[[ZERO]] : f16
// CHECK-NEXT:      arith.select %[[C]]
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
module {
  func.func @where_1d_device(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) attributes {grid = [1]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf16>
    %c128_i32 = arith.constant 128 : i32
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = ktdp.construct_memory_view %arg0, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %3 = ktdp.construct_memory_view %arg1, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %4 = ktdp.construct_memory_view %arg2, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %5 = ktdp.construct_memory_view %arg3, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %6 = ktdp.construct_memory_view %arg4, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %7 = ktdp.construct_memory_view %arg5, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>}} : memref<128xf16>
    %8 = arith.muli %1, %c128_i32 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = ktdp.construct_access_tile %2[%9] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    %11 = ktdp.load %10 : <128xindex> -> tensor<128xf16>
    %12 = arith.index_cast %8 : i32 to index
    %13 = ktdp.construct_access_tile %3[%12] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    %14 = ktdp.load %13 : <128xindex> -> tensor<128xf16>
    %15 = tensor.empty() : tensor<128xi1>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%11, %14 : tensor<128xf16>, tensor<128xf16>) outs(%15 : tensor<128xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: i1):
      %36 = arith.cmpf une, %in, %in_0 : f16
      linalg.yield %36 : i1
    } -> tensor<128xi1>
    %17 = tensor.empty() : tensor<128xf16>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%16 : tensor<128xi1>) outs(%17 : tensor<128xf16>) {
    ^bb0(%in: i1, %out: f16):
      %36 = arith.uitofp %in : i1 to f16
      linalg.yield %36 : f16
    } -> tensor<128xf16>
    %19 = arith.index_cast %8 : i32 to index
    %20 = ktdp.construct_access_tile %6[%19] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    ktdp.store %18, %20 : tensor<128xf16>, <128xindex>
    %21 = arith.index_cast %8 : i32 to index
    %22 = ktdp.construct_access_tile %6[%21] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    %23 = ktdp.load %22 : <128xindex> -> tensor<128xf16>
    %24 = arith.index_cast %8 : i32 to index
    %25 = ktdp.construct_access_tile %4[%24] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    %26 = ktdp.load %25 : <128xindex> -> tensor<128xf16>
    %27 = arith.index_cast %8 : i32 to index
    %28 = ktdp.construct_access_tile %5[%27] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    %29 = ktdp.load %28 : <128xindex> -> tensor<128xf16>
    %30 = tensor.empty() : tensor<128xi1>
    %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%23, %cst : tensor<128xf16>, tensor<128xf16>) outs(%30 : tensor<128xi1>) {
    ^bb0(%in: f16, %in_0: f16, %out: i1):
      %36 = arith.cmpf une, %in, %in_0 : f16
      linalg.yield %36 : i1
    } -> tensor<128xi1>
    %32 = tensor.empty() : tensor<128xf16>
    %33 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%31, %26, %29 : tensor<128xi1>, tensor<128xf16>, tensor<128xf16>) outs(%32 : tensor<128xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %36 = arith.select %in, %in_0, %in_1 : f16
      linalg.yield %36 : f16
    } -> tensor<128xf16>
    %34 = arith.index_cast %8 : i32 to index
    %35 = ktdp.construct_access_tile %7[%34] {access_tile_order = #map, access_tile_set = #set} : memref<128xf16> -> !ktdp.access_tile<128xindex>
    ktdp.store %33, %35 : tensor<128xf16>, <128xindex>
    return
  }
}
