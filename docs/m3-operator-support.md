# M3 general operator performance

## Split from correctness support

The MoE split OAI correctness patch is submitted separately as
[FlagGems #6916](https://github.com/flagos-ai/FlagGems/pull/6916). This PR does
not edit `fused_moe.py`, MoE activation parameters or quantization schedules.
Both PRs target the same upstream base and can be applied independently.

## API and contracts

- `flag_gems.swiglu_oai(x, limit=7.0, alpha=1.702, beta=1.0)` fuses the staged
  dense/shared split OAI chain. Every former eager intermediate rounds to the
  FP16/BF16 input dtype. Strided 2D and contiguous higher-rank inputs, empties,
  finite scalar parameters and NaN propagation are supported. This formula
  has different rounding boundaries from the one-round MoE activation in the
  separate correctness PR and must not replace it.
- `flag_gems.scaled_mm_int8(a, b, scale_a, scale_b, bias=None,
  out_dtype=torch.bfloat16)` is an explicit signed INT8 CUDA inference API
  with K<=65536, contiguous scalar/per-row/per-column FP32 scales, optional
  contiguous floating bias, FP16/BF16/FP32 output and strided matrices.
  Current upstream's ATen `scaled_mm` is FP8-only and remains unchanged.
  This explicit API retains the prior INT8 generic kernel and installed
  `scaled_mm` autotuner configurations without altering ATen's schema.
- The existing specialized column-major INT8 entrypoint retains priority.
  `FLAGGEMS_I8_SCALED_MM_SHAPE_TILES=1` selects the validated exact-shape SM90
  tiles only on the generic fallback. Other shapes keep autotuning; default
  is off. Benchmarks below use row-major B and compare the same generic
  kernel's installed autotuner against the fixed tile. They are not a claim
  of an improvement to the specialized entrypoint.

The fixed OAI schedule retains staged numerical boundaries; the exact-shape
tile switch is a documented tuning exception. Both new APIs use Triton and
metadata/empty allocations, with no Torch compute/copy/cast fallback.
Unsupported dtypes/layouts/devices and backward use are rejected explicitly.
Cross-backend INT8 execution has not been validated; the public API currently
accepts CUDA only.

## Validation

On an isolated H100, 43 numerical cases passed: 23 staged OAI cases and 20
INT8 cases covering the seven measured tile shapes, three output dtypes,
empty/tail/strided matrices, changed CUDA Graph inputs, specialized-route
priority, invalid scale/input dtypes, and unchanged FP8-only ATen validation.
The two repository benchmark entrypoints and standalone two-round Graph
script are checked separately. No serving speedup is inferred from these
synthetic measurements.

## Synthetic performance

25 warmup calls, 100 captured calls, five CUDA-event samples and two rounds
with reversed order. Values are mean round medians. The JSON preserves raw
samples and both rounds. OAI reference is the complete staged Torch chain;
INT8 reference is the same generic INT8 kernel with existing autotuning.

| Operator / shape | Baseline us | Candidate us | Speedup |
|---|---:|---:|---:|
| `swiglu_oai` [1, 384] | 9.448 | 1.120 | 8.437x |
| `swiglu_oai` [64, 768] | 12.293 | 1.496 | 8.219x |
| `swiglu_oai` [4096, 384] | 23.650 | 6.087 | 3.886x |
| `swiglu_oai` [5089, 768] | 54.868 | 10.424 | 5.263x |
| `swiglu_oai` [8192, 1536] | 176.393 | 31.917 | 5.527x |
| `scaled_mm_hopper_row_major_fallback` [1, 6144, 1536] | 23.577 | 21.101 | 1.117x |
| `scaled_mm_hopper_row_major_fallback` [64, 6144, 1536] | 23.828 | 20.724 | 1.150x |
| `scaled_mm_hopper_row_major_fallback` [4096, 6144, 1536] | 430.792 | 402.178 | 1.071x |
| `scaled_mm_hopper_row_major_fallback` [5089, 6144, 3072] | 1044.888 | 990.681 | 1.055x |
| `scaled_mm_hopper_row_major_fallback` [8192, 1024, 6144] | 596.975 | 570.162 | 1.047x |
| `scaled_mm_hopper_row_major_fallback` [4096, 1536, 6144] | 437.427 | 419.757 | 1.042x |
| `scaled_mm_hopper_row_major_fallback` [5089, 6144, 768] | 273.834 | 257.442 | 1.064x |

## Reproduce

```sh
VLLM_PLUGINS= PYTHONPATH=src python -m pytest -q tests/test_swiglu_oai.py tests/test_scaled_mm_int8.py
VLLM_PLUGINS= PYTHONPATH=src python -m pytest -q benchmark/test_swiglu_oai.py benchmark/test_scaled_mm_int8.py --level core --warmup 1 --iter 2
VLLM_PLUGINS= PYTHONPATH=src python benchmark/m3_graph_benchmark.py --output m3-operator-graph-results.json
```

For INT8 serving integration, callers must explicitly select
`scaled_mm_int8`; these PRs do not change any serving plugin dispatch.
