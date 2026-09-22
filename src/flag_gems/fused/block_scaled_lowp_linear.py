# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""E4M3 activations times E4M3/E2M1 weights using BF16 matrix arithmetic.

The per-32 activation quantization is retained, even on the compatibility
compute path. Each unscaled 32-wide product is accumulated in FP32 and then
scaled, matching the DeepSeek V4.1 reference's reduction contract. Weights
remain packed throughout; this is not a full-weight dequantization cache.

Semantic reference: DeepSeek-V4.1-Flash inference/kernel.py, revision
dba1be0a40aa45a94ad051997016db3960a90277, fp8_gemm / fp4_gemm.
This implementation uses no inline PTX or native FP8/FP4 dot instructions.
Only NVIDIA H100 execution has been validated so far.
"""

import torch
import triton
import triton.language as tl

from flag_gems.fused.act_quant import act_quant_triton


@triton.jit
def _e2m1(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 2,
        magnitude * 0.5,
        (1.0 + (magnitude & 1) * 0.5) * tl.exp2((magnitude >> 1) - 1.0),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _e8m0(code):
    # E8M0 has neither sign nor zero. Code 0 is the FP32 subnormal 2**-127;
    # code 255 is NaN, rather than an infinity or a finite scale.
    bits = tl.where(code == 0, 0x00400000, code.to(tl.uint32) << 23)
    return tl.where(code == 255, float("nan"), bits.to(tl.float32, bitcast=True))


@triton.jit
def _block_scaled_lowp_mm(
    A,
    AS,
    W,
    WS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WEIGHT_FP4: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 32):
        kk = group * 32 + offsets
        a = tl.load(
            A + rows[:, None] * K + kk[None, :], mask=rows[:, None] < M, other=0.0
        ).to(tl.bfloat16)
        if WEIGHT_FP4:
            packed = tl.load(
                W + cols[None, :] * (K // 2) + kk[:, None] // 2,
                mask=cols[None, :] < N,
                other=0,
            ).to(tl.uint8)
            codes = (packed >> ((kk[:, None] & 1) * 4)) & 15
            w = _e2m1(codes).to(tl.bfloat16)
            scale_rows = cols
        else:
            w = tl.load(
                W + cols[None, :] * K + kk[:, None], mask=cols[None, :] < N, other=0.0
            ).to(tl.bfloat16)
            scale_rows = cols // 32
        a_scale = tl.load(AS + rows * (K // 32) + group, mask=rows < M, other=0)
        w_code = tl.load(WS + scale_rows * (K // 32) + group, mask=cols < N, other=127)
        w_scale = _e8m0(w_code)
        product = tl.dot(a, w)
        acc = acc + product * a_scale[:, None] * w_scale[None, :]
    tl.store(
        OUT + rows[:, None] * N + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def block_scaled_lowp_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    weight_format: str,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Linear with per-32 E4M3 activation rounding and original E8M0 scales.

    ``weight_format='mxfp4'`` accepts [N,K/2] uint8, low nibble first, and
    [N,K/32] scales. ``'fp8'`` accepts [N,K] E4M3 and [ceil(N/32),K/32]
    scales. Scale storage is E8M0 or its uint8 byte view, never FP32 values
    accidentally reinterpreted as exponent bytes. Inputs must be finite.
    """
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] % 32:
        raise ValueError("Activation K must be positive and divisible by 32")
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_contiguous():
        raise ValueError("Expected contiguous BF16/FP16 activations")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported output dtype")
    if (
        weight.ndim != 2
        or not weight.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("Expected contiguous row-major weights and scales")
    if not (x.device == weight.device == weight_scale.device):
        raise ValueError("Activations, weights and scales must share a device")
    if weight_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("Expected E8M0 scale bytes")
    n, k = weight.shape[0], x.shape[-1]
    if weight_format == "mxfp4":
        valid = weight.dtype == torch.uint8 and weight.shape[1] == k // 2
        expected_scale = (n, k // 32)
    elif weight_format == "fp8":
        valid = weight.dtype == torch.float8_e4m3fn and weight.shape[1] == k
        expected_scale = (triton.cdiv(n, 32), k // 32)
    else:
        raise ValueError("weight_format must be mxfp4 or fp8")
    if not valid or weight_scale.shape != expected_scale:
        raise ValueError(
            "Weight/scale encoding or shape violates the quantization contract"
        )
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=output_dtype)
    m = x.numel() // k
    if m == 0 or n == 0:
        return out
    quantized, scales = act_quant_triton(x, block_size=32, scale_fmt="ue8m0")
    _block_scaled_lowp_mm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
        quantized,
        scales,
        weight,
        weight_scale.view(torch.uint8),
        out,
        m,
        n,
        k,
        WEIGHT_FP4=weight_format == "mxfp4",
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
