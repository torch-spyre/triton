// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// Note: tt.inter_tile_reduce uses a declarative assembly format.
// Semantic validation (unknown mode, scatter_dimension misuse, missing
// identities for region combiner) is performed by the LowerInterTile pass,
// not at parse time.  The tests below cover parse-level failures: malformed
// syntax that the declarative format parser rejects before any IR is built.

// ===------------------------------------------------------------------------===
// Missing "partials" keyword
// ===------------------------------------------------------------------------===
tt.func @missing_partials_keyword(%p: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %0 = tt.inter_tile_reduce
         // expected-error @+1 {{expected 'partials'}}
         (%p : tensor<1x64xf32>)
         identities(%p : tensor<1x64xf32>)
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<1x64xf32>)
  tt.return %0 : tensor<1x64xf32>
}

// -----

// ===------------------------------------------------------------------------===
// Missing "identities" keyword
// ===------------------------------------------------------------------------===
tt.func @missing_identities(%p: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<1x64xf32>)
         // expected-error @+1 {{expected 'identities'}}
         axis = "x" mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<1x64xf32>)
  tt.return %0 : tensor<1x64xf32>
}

// -----

// ===------------------------------------------------------------------------===
// Missing "axis" keyword
// ===------------------------------------------------------------------------===
tt.func @missing_axis(%p: tensor<1x64xf32>, %id: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<1x64xf32>)
         identities(%id : tensor<1x64xf32>)
         // expected-error @+1 {{expected 'axis'}}
         mode = "all_reduce" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<1x64xf32>)
  tt.return %0 : tensor<1x64xf32>
}

// -----

// ===------------------------------------------------------------------------===
// Missing "mode" keyword
// ===------------------------------------------------------------------------===
tt.func @missing_mode(%p: tensor<1x64xf32>, %id: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<1x64xf32>)
         identities(%id : tensor<1x64xf32>)
         // expected-error @+1 {{expected 'mode'}}
         axis = "x" combiner = "add"
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<1x64xf32>)
  tt.return %0 : tensor<1x64xf32>
}

// -----

// ===------------------------------------------------------------------------===
// Missing "combiner" keyword
// ===------------------------------------------------------------------------===
tt.func @missing_combiner(%p: tensor<1x64xf32>, %id: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %0 = tt.inter_tile_reduce
         partials(%p : tensor<1x64xf32>)
         identities(%id : tensor<1x64xf32>)
         axis = "x" mode = "all_reduce"
         // expected-error @+1 {{expected 'combiner'}}
         {numWkSlicesPerDim = {x = 2 : i64},
          coreIdToWkSlice = [{x = 0 : i64}, {x = 1 : i64}]}
         -> (tensor<1x64xf32>)
  tt.return %0 : tensor<1x64xf32>
}
