// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic=data-layout=hsot -verify-diagnostics

// An unrecognized data-layout must be rejected. The pass resolves the option as
// `hwDataLayout = (dataLayout == "device")`, so without this check a typo would
// silently select the "host" layout instead of failing. The pass is invocable
// directly, bypassing the frontend's own validation, so the check has to be
// here.

// expected-error @below {{rewrite-descriptor-layout-generic: data-layout must be 'device' or 'host', got 'hsot'}}
module {
tt.func @typo_in_data_layout() {
  tt.return
}
}
