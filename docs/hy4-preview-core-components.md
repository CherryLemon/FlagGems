# Hy4-preview component specializations

This change delivers two existing component implementations through Core
FlagGems. It does not establish a full-model numeric or serving speedup claim.

## Row-padded RMSNorm

The CUDA forward wrapper passes a row-padded input directly to the existing
small RMSNorm kernel when it is a two-dimensional FP16/BF16/FP32 strided tensor,
has one normalized dimension with 0 < N <= 4096, unit column stride, and
non-overlapping positive rows. Other layouts retain the existing path. All
numerical kernels, backward functions and their schedules remain unchanged.

The archived candidate source SHA256 was
`1d6b94ec6052eba20473d39380d1d462b29f09b9fdc4be3071193b6e77ef8206`.
On H100, Torch2.13/cu130 and FlagTree3.6, 26 finite cases passed exact eager
and ordinary CUDA graph A/B/A comparisons against the original packed-input
forward, including N512/2048, M1/4/64/256/1024, nonzero offsets, odd row stride,
FP16/BF16/FP32, and fallback cases. These are component tests, not TP16 model
qualification. The port preserves the qualified forward AST and all current
upstream mathematical kernel ASTs.

`benchmark/test_rms_norm.py::test_rms_norm_row_padding` compares complete forward
calls against the same kernel with its previous required input materialization.
The M1 contiguous-view case is retained as a control. No new schedule or autotune
space is introduced.

## Small-M BF16 linear

The opt-in `flag_gems.fused.bf16_small_m_linear` API accepts contiguous CUDA BF16
input [M,6144] and weight [1024,6144] for 1 <= M <= 16 on SM90. The caller passes
the device capability resolved before capture as `device_sm=90`. Bias, other
devices, layouts and shapes are rejected. This is a forward-only inference API;
call with gradients disabled. It does not register a replacement for aten linear
or select a serving-framework provider.

The archived source SHA256 was
`bcbada2ea37577393fef32b41e51989dc9312eec0cef41a27d3a56fe6dfe9260`.
H100 component runs passed strict projection and final-gate byte comparisons to
the original FlagGems path for M1/4 and the later M2/8/16 complement, including
ordinary fixed-address A/B/A graph replay. M64/256/1024 exercised the caller's
existing fallback and did not select this API. Inputs and weights were synthetic
at actual module shapes, not checkpoint weights.

The fixed BM16/BN16/BK256, four-warp/four-stage schedule is an explicit tuning
exemption for this bounded specialization. It preserves the qualified reduction
tree; any new schedule requires separate numeric qualification. Tests and the
standalone benchmark are provided. The current-upstream GPU tests and benchmark
have not been rerun during this CPU-only delivery preparation.

## Scope and evidence limits

These are two historical contracts: one host-dispatch change (historically
classified as other) and one tuned numerical specialization. They add zero
newly authored numerical kernels in this delivery round. The small-M function is one newly exported Core
API for an existing semantic operation. Neither source count nor component
timing is an operator-coverage denominator. Formal runtime coverage is unknown.

The INT8 indexer eager fixtures, unqualified RoPE branches, foreach integration,
and stable-topk performance experiments are separate work. No source-lifetime,
owner-deletion or alias-replacement experiment is included in these tests.
