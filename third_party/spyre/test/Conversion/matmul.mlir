// CHECK: module
// CHECK-LABEL: func.func @matmul_f16
// CHECK-SAME:  %[[A:[^ ]*]]: tensor<16x32xf16>{{.*}}%[[B:[^ ]*]]: tensor<32x8xf16>{{.*}}%[[C:[^ ]*]]: tensor<16x8xf32>{{.*}}-> tensor<16x8xf32>
module {
tt.func @matmul_f16(%a: tensor<16x32xf16>, %b: tensor<32x8xf16>, %c: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NOT:  tt.dot
  // CHECK:      %[[RES:[^ ]*]] = linalg.matmul
  // CHECK-SAME:   ins(%[[A]], %[[B]] : tensor<16x32xf16>, tensor<32x8xf16>)
  // CHECK-SAME:   outs(%[[C]] : tensor<16x8xf32>)
  // CHECK-SAME:   -> tensor<16x8xf32>
  // CHECK-NOT:  tt.dot
  %0 = tt.dot %a, %b, %c : tensor<16x32xf16> * tensor<32x8xf16> -> tensor<16x8xf32>

  // CHECK:      return %[[RES]] : tensor<16x8xf32>
  tt.return %0 : tensor<16x8xf32>
}
}
