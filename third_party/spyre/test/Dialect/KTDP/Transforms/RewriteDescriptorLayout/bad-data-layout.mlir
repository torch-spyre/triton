// RUN: spyre-triton-opt %s --rewrite-descriptor-layout=data-layout=hsot -verify-diagnostics

// An unrecognized data-layout must be rejected. The pass resolves the option
// as `hwDataLayout = (dataLayout == "device")`, so without this check a typo
// would silently select the "host" layout instead of failing.

// expected-error @below {{rewrite-descriptor-layout: data-layout must be 'device' or 'host', got 'hsot'}}
module {
tt.func @bad_data_layout(%ptr: !tt.ptr<f32>) {
  tt.return
}
}
