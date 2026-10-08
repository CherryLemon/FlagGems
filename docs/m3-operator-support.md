# M3 general operator changes

## API and scope

- `flag_gems.swiglu_oai(x, limit=7.0, alpha=1.702, beta=1.0)` is a forward inference primitive for split `[gate..., up...]` FP16/BF16 activations. Gate clamps above `limit`; up clamps to ±limit. Every former eager intermediate rounds to the input dtype. It supports strided 2D and contiguous higher-rank inputs, empty dimensions, finite scalar parameters, and NaN propagation. Backward and other dtypes/layouts are explicitly unsupported.
- `FLAGGEMS_I8_SCALED_MM_SHAPE_TILES=1` selects opt-in INT8 tiles for the validated SM90 shape set. Uncovered shapes retain the autotuner. This applies to the generic Triton fallback; the existing specialized scaled-MM entrypoint retains priority. Default dispatch is unchanged.

Fixed OAI launches preserve the validated staged arithmetic. Existing GEMM tuning remains available for uncovered inputs; the exact-shape opt-in is the stated tuning exception. There is no new Torch compute fallback.

## Validation

H100: 23 OAI tests and seven INT8 fallback tests passed. Coverage includes both OAI dtypes, parameter variants, odd intermediate widths, strided inputs, empties, NaN/Inf, explicit unsupported paths, changed CUDA Graph inputs, and BF16 output with bias/scales for all tile configurations. Autograd is not provided.

## Synthetic performance

CUDA Graph timing: 25 warmup calls, 100 captured calls, five CUDA-event samples per round, and two rounds with reversed baseline/candidate order. Values below are the mean of the two round medians; JSON preserves each sample and each round. OAI baseline is the complete staged Torch chain. INT8 baseline is the same generic Triton kernel through its existing autotuner, using row-major B. No checkpoint or model throughput data is used.

| Operator / shape | Baseline us | Candidate us | Speedup |
|---|---:|---:|---:|
| `swiglu_oai` [1, 384] | 9.450 | 1.128 | 8.378x |
| `swiglu_oai` [64, 768] | 12.327 | 1.478 | 8.341x |
| `swiglu_oai` [4096, 384] | 23.632 | 6.086 | 3.883x |
| `swiglu_oai` [5089, 768] | 54.781 | 10.441 | 5.246x |
| `swiglu_oai` [8192, 1536] | 176.188 | 31.887 | 5.525x |
| `scaled_mm_hopper_row_major_fallback` [1, 6144, 1536] | 23.564 | 21.044 | 1.120x |
| `scaled_mm_hopper_row_major_fallback` [64, 6144, 1536] | 23.834 | 20.680 | 1.153x |
| `scaled_mm_hopper_row_major_fallback` [4096, 6144, 1536] | 430.839 | 402.177 | 1.071x |
| `scaled_mm_hopper_row_major_fallback` [5089, 6144, 3072] | 1044.501 | 990.912 | 1.054x |
| `scaled_mm_hopper_row_major_fallback` [8192, 1024, 6144] | 597.014 | 570.095 | 1.047x |
| `scaled_mm_hopper_row_major_fallback` [4096, 1536, 6144] | 437.488 | 419.731 | 1.042x |
| `scaled_mm_hopper_row_major_fallback` [5089, 6144, 768] | 273.730 | 257.377 | 1.064x |

For the normal transposed-weight specialized entrypoint, the opt-in is not selected: a separate check measured approximately 1.00x, as expected. The fallback result must not be claimed as a speedup for that entrypoint or as end-to-end serving performance.

Reproduce numerical tests:

```sh
PYTHONPATH=src python -m pytest -q tests/test_swiglu_oai.py
PYTHONPATH=src python -m pytest -q tests/test_scaled_mm.py -k hopper_opt_in
```

Repository benchmarks include `benchmark/test_swiglu_oai.py` and the `test_int8_hopper_fallback` case in `benchmark/test_scaled_mm.py`. JSON above uses the explicit Graph protocol, so it is separate from the repository benchmark's default timing mode. Non-Hopper tile performance has not been validated; the opt-in is gated to SM90.

The exact Graph timing protocol is runnable without serving artifacts:

```sh
VLLM_PLUGINS= PYTHONPATH=src python benchmark/m3_graph_benchmark.py --output m3-operator-graph-results.json
```
