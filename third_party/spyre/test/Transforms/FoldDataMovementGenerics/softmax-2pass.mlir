// RUN: spyre-triton-opt %s --convert-elementwise-to-linalg --linalg-generalize-named-ops --unalias-linalg-outs --fold-data-movement-generics 2>&1 | FileCheck %s

// REAL IR, not a hand-written shape: the `ktir`-stage output of the
// `softmax_2pass` fixture verbatim, locations stripped. Its own file because it is
// the only case here that is a whole kernel, and the only one needing the three
// passes ahead of this one in the pipeline.
//
// Two things at once. As a POSITIVE it is the absorber on a kernel nobody wrote
// for it: 2 `tensor.expand_shape` and 3 `tensor.collapse_shape`, all absorbed,
// none surviving -- and the hardcoded absorber this replaced could not have taken
// the two expands. As an OVER-INCLUSION check it carries zero
// `tts.tensor_layout`, so the gate must stay shut over all five; a gate keyed on
// the shape ops rather than on the path would refuse the whole kernel.
//
// The RUN line needs the other three passes because they are what puts the shape
// ops in front of generics: at the `ktir` stage the computes are still `arith.*`
// on tensors and the broadcasts and reduces are still named linalg ops, and
// fusion matches generic -> generic only.

// CHECK-NOT: error
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK: func.func @softmax_2pass
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  func.func @softmax_2pass(%arg0: index, %arg1: index, %arg2: i32, %arg3: i32) attributes {grid = [32]} {
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c32_i32 = arith.constant 32 : i32
    %c3_i32 = arith.constant 3 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x1xf32>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<4x1xf32>
    %c4_i32 = arith.constant 4 : i32
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = ktdp.construct_memory_view %arg1, sizes: [%2, %3], strides: [%4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<?x?xf16>
    %6 = arith.index_cast %arg2 : i32 to index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = ktdp.construct_memory_view %arg0, sizes: [%6, %7], strides: [%8, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<?x?xf16>
    %10 = arith.addi %arg2, %c31_i32 : i32
    %11 = arith.divsi %10, %c32_i32 : i32
    %12 = arith.muli %1, %11 : i32
    %13 = arith.addi %12, %11 : i32
    %14 = arith.minsi %13, %arg2 : i32
    %15 = arith.addi %arg3, %c63_i32 : i32
    %16 = arith.divsi %15, %c64_i32 : i32
    %17 = arith.subi %14, %12 : i32
    %18 = arith.addi %17, %c3_i32 : i32
    %19 = arith.divsi %18, %c4_i32 : i32
    scf.for %arg4 = %c0_i32 to %19 step %c1_i32  : i32 {
      %20 = arith.muli %arg4, %c4_i32 : i32
      %21 = arith.addi %12, %20 : i32
      %22:2 = scf.for %arg5 = %c0_i32 to %16 step %c1_i32 iter_args(%arg6 = %cst_2, %arg7 = %cst_1) -> (tensor<4x1xf32>, tensor<4x1xf32>)  : i32 {
        %23 = arith.muli %arg5, %c64_i32 : i32
        %24 = arith.index_cast %21 : i32 to index
        %25 = arith.index_cast %23 : i32 to index
        %26 = ktdp.construct_access_tile %5[%24, %25] {access_tile_order = #map, access_tile_set = #set1} : memref<?x?xf16> -> !ktdp.access_tile<4x64xindex>
        %27 = ktdp.load %26 : <4x64xindex> -> tensor<4x64xf16>
        %28 = arith.extf %27 : tensor<4x64xf16> to tensor<4x64xf32>
        %29 = tensor.empty() : tensor<4xf32>
        %30 = linalg.fill ins(%cst_0 : f32) outs(%29 : tensor<4xf32>) -> tensor<4xf32>
        %reduced = linalg.reduce ins(%28 : tensor<4x64xf32>) outs(%30 : tensor<4xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %41 = arith.maxnumf %in, %init : f32
            linalg.yield %41 : f32
          }
        %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
        %31 = arith.maxnumf %arg6, %expanded : tensor<4x1xf32>
        %32 = arith.subf %arg6, %31 : tensor<4x1xf32>
        %33 = math.exp %32 : tensor<4x1xf32>
        %34 = arith.mulf %arg7, %33 : tensor<4x1xf32>
        %collapsed = tensor.collapse_shape %31 [[0, 1]] : tensor<4x1xf32> into tensor<4xf32>
        %35 = tensor.empty() : tensor<4x64xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<4xf32>) outs(%35 : tensor<4x64xf32>) dimensions = [1] 
        %36 = arith.subf %28, %broadcasted : tensor<4x64xf32>
        %37 = math.exp %36 : tensor<4x64xf32>
        %38 = tensor.empty() : tensor<4xf32>
        %39 = linalg.fill ins(%cst : f32) outs(%38 : tensor<4xf32>) -> tensor<4xf32>
        %reduced_3 = linalg.reduce ins(%37 : tensor<4x64xf32>) outs(%39 : tensor<4xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %41 = arith.addf %in, %init : f32
            linalg.yield %41 : f32
          }
        %expanded_4 = tensor.expand_shape %reduced_3 [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
        %40 = arith.addf %34, %expanded_4 : tensor<4x1xf32>
        scf.yield %31, %40 : tensor<4x1xf32>, tensor<4x1xf32>
      }
      scf.for %arg5 = %c0_i32 to %16 step %c1_i32  : i32 {
        %23 = arith.muli %arg5, %c64_i32 : i32
        %24 = arith.index_cast %21 : i32 to index
        %25 = arith.index_cast %23 : i32 to index
        %26 = ktdp.construct_access_tile %5[%24, %25] {access_tile_order = #map, access_tile_set = #set1} : memref<?x?xf16> -> !ktdp.access_tile<4x64xindex>
        %27 = ktdp.load %26 : <4x64xindex> -> tensor<4x64xf16>
        %28 = arith.extf %27 : tensor<4x64xf16> to tensor<4x64xf32>
        %collapsed = tensor.collapse_shape %22#0 [[0, 1]] : tensor<4x1xf32> into tensor<4xf32>
        %29 = tensor.empty() : tensor<4x64xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<4xf32>) outs(%29 : tensor<4x64xf32>) dimensions = [1] 
        %30 = arith.subf %28, %broadcasted : tensor<4x64xf32>
        %31 = math.exp %30 : tensor<4x64xf32>
        %collapsed_3 = tensor.collapse_shape %22#1 [[0, 1]] : tensor<4x1xf32> into tensor<4xf32>
        %32 = tensor.empty() : tensor<4x64xf32>
        %broadcasted_4 = linalg.broadcast ins(%collapsed_3 : tensor<4xf32>) outs(%32 : tensor<4x64xf32>) dimensions = [1] 
        %33 = arith.divf %31, %broadcasted_4 : tensor<4x64xf32>
        %34 = arith.truncf %33 : tensor<4x64xf32> to tensor<4x64xf16>
        %35 = arith.index_cast %21 : i32 to index
        %36 = arith.index_cast %23 : i32 to index
        %37 = ktdp.construct_access_tile %9[%35, %36] {access_tile_order = #map, access_tile_set = #set1} : memref<?x?xf16> -> !ktdp.access_tile<4x64xindex>
        ktdp.store %34, %37 : tensor<4x64xf16>, <4x64xindex>
      }
    }
    return
  }
}

