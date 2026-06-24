# Spyre backend tests

The Spyre backend test suite uses **two complementary test frameworks**, each
suited to a different granularity of coverage:

- **Structural tests** — Python assertions over the produced IR, driven by
  `SinglePassTester` (see `test_lower_compute_ops.py`, `test_lower_desc_memory.py`,
  `test_distribute_work.py`). One pass is run in isolation and the resulting ops
  are inspected programmatically (`assert_present`, `assert_absent`,
  `assert_result_type`, …).
- **FileCheck tests** — LLVM `FileCheck` directives embedded in `.mlir` inputs
  under `Conversion/`, driven by `test_conversion.py` and the `check_ir` fixture
  in `conftest.py`. The full TTIR → KTIR pipeline is run and the entire IR text
  is matched against `// CHECK*` lines.

## Structural tests are the right tool for per-pass coverage

Structural tests target **one pass at a time**. A test subclasses
`SinglePassTester`, sets `PASS` to the pass-adder (e.g.
`PASS = "add_lower_compute_ops"`), feeds in a small TTIR module, and asserts on
the ops the pass produced or removed:

```python
class TestSplat(LowerComputeOpsTester):
    def test_f32_1d(self):
        self.run("""
        module {
          tt.func @k(%s: f32) -> tensor<1024xf32> {
            %0 = tt.splat %s : f32 -> tensor<1024xf32>
            tt.return %0 : tensor<1024xf32>
          }
        }
        """)
        self.assert_present("linalg.fill")
        self.assert_absent("tt.splat")
        self.assert_result_type("linalg.fill", "tensor<1024xf32>")
```

This makes them the right tool for **per-pass coverage**:

- They isolate a single transform, so a failure points directly at the pass
  under test rather than at some interaction further down the pipeline.
- They cover shape/type variants and negative (expected-failure) cases
  cheaply — negative tests use `pytest.raises` plus a stderr diagnostic check.
- They are easy to read and maintain alongside the pass they exercise (one test
  class per Triton op), and integrate with the `@pattern` decorator to generate
  per-pattern documentation.

## FileCheck tests are for integration tests

FileCheck tests target the **whole pipeline**. Each input under `Conversion/` is
a `.mlir` file that carries its own `// CHECK*` directives; the full TTIR → KTIR
lowering is run and the produced IR is piped to LLVM's `FileCheck`:

```mlir
// CHECK-LABEL: func.func @matmul_f16
module {
tt.func @matmul_f16(%a: tensor<16x32xf16>, %b: tensor<32x8xf16>, %c: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NOT:  tt.dot
  // CHECK:      %[[RES:[^ ]*]] = linalg.matmul
  %0 = tt.dot %a, %b, %c : tensor<16x32xf16> * tensor<32x8xf16> -> tensor<16x8xf32>
  tt.return %0 : tensor<16x8xf32>
}
}
```

This makes them the right tool for **integration tests**:

- They exercise the passes together end-to-end, catching interaction and
  ordering problems that per-pass tests cannot see.
- The expected IR lives next to the input as `CHECK` lines, so the test reads
  like a golden transcript of the lowering, including variable bindings
  (`%[[RES]]`), `CHECK-SAME`, and `CHECK-NOT` constraints.
- They use the same `FileCheck` semantics (`--enable-var-scope`) as the upstream
  Triton lit tests, so authoring is familiar to anyone who has written MLIR
  conversion tests.

`FileCheck` is located via `FILECHECK_PATH`, a generated `lit.site.cfg.py`, or
the build tree; if it cannot be found the FileCheck tests skip rather than fail.

## Future work

- **Enable `lit` integration.** Today the FileCheck tests are driven through
  pytest (`test_conversion.py` + the `check_ir` fixture). The goal is to run the
  `Conversion/` directory under `lit` directly, reusing the upstream Triton lit
  configuration so the tests behave identically to the rest of the Triton test
  tree.
- **Improve the FileCheck tests for `lit`.** Expand the `Conversion/` corpus and
  refine the `CHECK` directives (prefixes, `RUN` lines, per-stage checks) so the
  files are self-contained lit tests.
- **Use one command or two commands** Run every tests from pytest or from lit test
  and pytest.
