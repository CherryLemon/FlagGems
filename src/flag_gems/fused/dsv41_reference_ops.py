# SPDX-License-Identifier: Apache-2.0
"""Portable low-precision cache rounding and sparse attention with a sink.

Contracts follow DeepSeek-V4.1-Flash's published inference/kernel.py (MIT).
Unlike a causal attention API, sparse positions here already encode visibility;
there is no comparison between an index and the query row number.
"""

import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim


@triton.jit
def _hc_sum4(x, axis: tl.constexpr):
    # Butterfly order, matching the published four-way FP32 reduction.
    lanes = tl.arange(0, 4)
    if axis == 1:
        p2 = tl.broadcast_to((lanes ^ 2)[None, :], (4, 4))
        p1 = tl.broadcast_to((lanes ^ 1)[None, :], (4, 4))
    else:
        p2 = tl.broadcast_to((lanes ^ 2)[:, None], (4, 4))
        p1 = tl.broadcast_to((lanes ^ 1)[:, None], (4, 4))
    pair = x + tl.gather(x, p2, axis)
    return pair + tl.gather(pair, p1, axis)


@triton.jit
def _hc_reference_kernel(
    X, S, B, Pre, Post, Comb, ITERS: tl.constexpr, EPS: tl.constexpr
):
    token = tl.program_id(0)
    cols = tl.arange(0, 4)
    s0, s1, s2 = tl.load(S), tl.load(S + 1), tl.load(S + 2)
    pre = tl.load(X + token * 24 + cols) * s0 + tl.load(B + cols)
    post = tl.load(X + token * 24 + cols + 4) * s1 + tl.load(B + cols + 4)
    pre = tl.div_rn(1.0, 1.0 + tl_extra_shim.exp(-pre)) + EPS
    post = 2.0 * tl.div_rn(1.0, 1.0 + tl_extra_shim.exp(-post))
    tl.store(Pre + token * 4 + cols, pre)
    tl.store(Post + token * 4 + cols, post)
    offsets = cols[:, None] * 4 + cols[None, :]
    comb = tl.load(X + token * 24 + offsets + 8) * s2 + tl.load(B + offsets + 8)
    comb = tl_extra_shim.exp(comb - tl.max(comb, 1)[:, None])
    comb = tl.div_rn(comb, _hc_sum4(comb, 1)) + EPS
    comb = tl.div_rn(comb, _hc_sum4(comb, 0) + EPS)
    for _ in range(ITERS - 1):
        comb = tl.div_rn(comb, _hc_sum4(comb, 1) + EPS)
        comb = tl.div_rn(comb, _hc_sum4(comb, 0) + EPS)
    tl.store(Comb + token * 16 + offsets, comb)


def hc_split_sinkhorn_reference(
    mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
):
    """FP32 four-stream mHC with precise exp/div and a fixed reduction tree.

    The published graph is sensitive to sub-ULP changes in these coefficients:
    BF16 residual rounding and expert selection can amplify them over 40 layers.
    This compatibility operator deliberately prioritizes that numerical contract.
    """
    if (
        hc_mult != 4
        or mixes.shape[-1] != 24
        or scale.shape != (3,)
        or base.shape != (24,)
    ):
        raise ValueError("expected four-stream mHC mixes[...,24], scale[3], base[24]")
    if sinkhorn_iters < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("positive iterations and finite positive epsilon are required")
    if any(
        t.dtype != torch.float32 or t.device != mixes.device
        for t in (mixes, scale, base)
    ):
        raise ValueError("all mHC inputs must be FP32 on the same device")
    if not all(t.is_contiguous() for t in (mixes, scale, base)):
        raise ValueError("mHC inputs must be contiguous")
    shape = mixes.shape[:-1]
    pre = torch.empty((*shape, 4), device=mixes.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((*shape, 4, 4), device=mixes.device, dtype=torch.float32)
    if mixes.numel():
        _hc_reference_kernel[(mixes.numel() // 24,)](
            mixes,
            scale,
            base,
            pre,
            post,
            comb,
            sinkhorn_iters,
            eps,
            num_warps=1,
            enable_fp_fusion=True,
        )
    return pre, post, comb


@libentry()
@triton.jit
def _fp4_round_kernel(
    X,
    Y,
    S,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    E4_SCALE: tl.constexpr,
    INPLACE: tl.constexpr,
):
    group = tl.program_id(0)
    col = tl.arange(0, GROUP)
    x = tl.load(X + group * GROUP + col).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    if E4_SCALE:
        scale = (
            (tl.maximum(amax, 6.0 * (2.0**-9)) / 6.0).to(tl.float8e4nv).to(tl.float32)
        )
        tl.store(S + group, scale)
    else:
        scaled = tl.maximum(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        bits = scaled.to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        tl.store(S + group, exponent.to(tl.uint8))
    value = tl.div_rn(x, scale)
    mag = tl.abs(value)
    # E2M1 round-to-nearest, ties-to-even. The nonuniform thresholds matter.
    code = tl.where(
        mag <= 0.25,
        0,
        tl.where(
            mag < 0.75,
            1,
            tl.where(
                mag <= 1.25,
                2,
                tl.where(
                    mag < 1.75,
                    3,
                    tl.where(
                        mag <= 2.5,
                        4,
                        tl.where(mag < 3.5, 5, tl.where(mag <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    )
    sign = ((value.to(tl.uint32, bitcast=True) >> 31) << 3).to(tl.int32)
    if INPLACE:
        decoded = tl.where(
            code < 4,
            code.to(tl.float32) * 0.5,
            tl.where(
                code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))
            ),
        )
        decoded = tl.where(sign != 0, -decoded, decoded)
        tl.store(Y + group * GROUP + col, decoded * scale)
    else:
        codes = code | sign
        pair = tl.reshape(codes, (GROUP // 2, 2))
        low, high = tl.split(pair)
        tl.store(Y + group * (GROUP // 2) + tl.arange(0, GROUP // 2), low | (high << 4))


def fp4_quantize_reference(x, group_size=32, *, scale_format="e8m0", inplace=False):
    """E2M1 packed codes/scales, or quantize-dequantize back to x's dtype.

    In-place mode accepts strided tensors and copies the rounded result back.
    Packed mode returns uint8 (two consecutive K values per byte) and typed
    scales. E4M3 scales are used by compressed KV, E8M0 by the indexer.
    """
    if x.dtype not in (torch.bfloat16, torch.float16) or x.ndim < 1:
        raise ValueError("x must be BF16/FP16 with at least one dimension")
    if group_size not in (16, 32) or x.shape[-1] % group_size:
        raise ValueError("complete groups of 16 or 32 are required")
    if scale_format not in ("e8m0", "e4m3"):
        raise ValueError("scale_format must be e8m0 or e4m3")
    z = x.contiguous()
    y = (
        torch.empty_like(z)
        if inplace
        else torch.empty(
            (*z.shape[:-1], z.shape[-1] // 2), dtype=torch.uint8, device=z.device
        )
    )
    scales = torch.empty(
        (*z.shape[:-1], z.shape[-1] // group_size),
        dtype=torch.float8_e4m3fn if scale_format == "e4m3" else torch.uint8,
        device=z.device,
    )
    if z.numel():
        _fp4_round_kernel[(z.numel() // group_size,)](
            z,
            y,
            scales,
            z.shape[-1],
            group_size,
            scale_format == "e4m3",
            inplace,
            num_warps=1,
            enable_fp_fusion=False,
        )
    if inplace:
        x.copy_(y)
        return x
    if scale_format == "e8m0":
        scales = scales.view(torch.float8_e8m0fnu)
    return y, scales


@libentry()
@triton.jit
def _sparse_sink_kernel(
    Q,
    KV,
    Sink,
    Idx,
    Out,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    Compressed,
    Pages,
    PAGED: tl.constexpr,
    WINDOW_STRIDE: tl.constexpr,
    COMPRESSED_STRIDE: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
):
    token, batch = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, BH)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + ((batch * S + token) * H + heads[:, None]) * D + dims[None, :],
        heads[:, None] < H,
        0.0,
    )
    acc = tl.full((BH, D), 0, tl.float32)
    maxima = tl.full((BH,), -1.0e30, tl.float32)
    total = tl.full((BH,), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        slots = start * BK + tl.arange(0, BK)
        ids = tl.load(Idx + (batch * S + token) * K + slots, slots < K, -1)
        if PAGED:
            page = tl.load(Pages + batch)
            valid = (ids >= 0) & (ids < N + COMPRESSED_N)
            window = tl.load(
                KV + page * WINDOW_STRIDE + ids[:, None] * D + dims[None, :],
                ((ids >= 0) & (ids < N))[:, None],
                0.0,
            )
            compressed = tl.load(
                Compressed
                + page * COMPRESSED_STRIDE
                + (ids[:, None] - N) * D
                + dims[None, :],
                ((ids >= N) & (ids < N + COMPRESSED_N))[:, None],
                0.0,
            )
            kv = tl.where((ids < N)[:, None], window, compressed)
        else:
            valid = (ids >= 0) & (ids < N)
            kv = tl.load(
                KV + (batch * N + ids[:, None]) * D + dims[None, :],
                valid[:, None],
                0.0,
            )
        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
        scores = tl.where(valid[None, :], scores, -float("inf"))
        new_max = tl.maximum(maxima, tl.max(scores, 1))
        rescale = tl_extra_shim.exp(maxima - new_max)
        prob = tl_extra_shim.exp(scores - new_max[:, None])
        if BH == 16:
            # The 16-head reference sums adjacent pairs, then eight column
            # groups, then four pairs. Preserve FP32 association before BF16 PV.
            block_sum = tl.sum(tl.sum(tl.sum(prob.reshape(BH, 8, 4, 2), 3), 1), 1)
        else:
            block_sum = tl.sum(prob, 1)
        total = total * rescale + block_sum
        acc = acc * rescale[:, None]
        acc = tl.dot(prob.to(q.dtype), kv, acc)
        maxima = new_max
    sink = tl.load(Sink + heads, heads < H, 0.0).to(tl.float32)
    denom = total + tl_extra_shim.exp(sink - maxima)
    result = tl.where(denom[:, None] > 0, tl.div_rn(acc, denom[:, None]), 0.0)
    tl.store(
        Out + ((batch * S + token) * H + heads[:, None]) * D + dims[None, :],
        result,
        heads[:, None] < H,
    )


def sparse_attention_with_sink(q, kv, attn_sink, indices, softmax_scale):
    """Sparse shared-KV attention, explicit visibility, BF16 PV rounding.

    q=[B,S,H,D], kv=[B,N,D], indices=[B,S,K] int32; -1 is an empty
    position. The learned per-head sink contributes only to the denominator.
    Workspace is independent of N*K and contains no dense attention matrix.
    """
    if q.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected q[B,S,H,D], kv[B,N,D], indices[B,S,K]")
    b, s, h, d = q.shape
    if (
        kv.shape[0] != b
        or kv.shape[-1] != d
        or indices.shape[:2] != (b, s)
        or attn_sink.shape != (h,)
        or d not in (64, 128, 256, 512)
    ):
        raise ValueError("incompatible sparse attention shapes")
    if q.dtype != torch.bfloat16 or kv.dtype != q.dtype or indices.dtype != torch.int32:
        raise ValueError("the reference contract requires BF16 Q/KV and int32 indices")
    if (
        attn_sink.dtype != torch.float32
        or not math.isfinite(softmax_scale)
        or softmax_scale <= 0
    ):
        raise ValueError("expected FP32 sink and positive finite softmax scale")
    if any(t.device != q.device for t in (kv, attn_sink, indices)):
        raise ValueError("all inputs must share a device")
    if not all(t.is_contiguous() for t in (q, kv, attn_sink, indices)):
        raise ValueError("all inputs must be contiguous")
    if h > 64:
        raise ValueError("at most 64 local heads are supported")
    out = torch.empty_like(q)
    if b * s:
        _sparse_sink_kernel[(s, b)](
            q,
            kv,
            attn_sink,
            indices,
            out,
            s,
            h,
            d,
            kv.shape[1],
            indices.shape[2],
            softmax_scale,
            max(16, triton.next_power_of_2(h)),
            64,
            kv,
            indices,
            False,
            0,
            0,
            0,
            num_stages=1,
            num_warps=8 if h >= 32 or d == 512 else 4,
            enable_fp_fusion=True,
        )
    return out


def paged_sparse_attention_with_sink(
    q, window, compressed, attn_sink, indices, pages, softmax_scale
):
    """Read selected request pages directly, without concatenating full KV.

    Window and compressed pools are [P,N,D], contiguous within a page; the
    page stride can include other state fields. Indices address their logical
    concatenation. Runtime page IDs are int64[B] and may change during replay.
    Visibility (including padded queries) is encoded by -1 indices.
    """
    if q.ndim != 4 or window.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected q[B,S,H,D], window[P,W,D], indices[B,S,K]")
    b, s, h, d = q.shape
    if compressed is None:
        compressed, compressed_n = window, 0
    else:
        compressed_n = compressed.shape[1]
    if (
        compressed.ndim != 3
        or window.shape[0] != compressed.shape[0]
        or window.shape[2] != d
        or compressed.shape[2] != d
        or pages.shape != (b,)
        or pages.dtype != torch.int64
        or indices.shape[:2] != (b, s)
        or indices.dtype != torch.int32
        or attn_sink.shape != (h,)
        or attn_sink.dtype != torch.float32
        or q.dtype != torch.bfloat16
        or window.dtype != q.dtype
        or compressed.dtype != q.dtype
        or d not in (64, 128, 256, 512)
        or not 0 < h <= 64
        or not math.isfinite(softmax_scale)
        or softmax_scale <= 0
    ):
        raise ValueError("incompatible paged sparse attention inputs")
    if any(
        t.device != q.device for t in (window, compressed, indices, pages, attn_sink)
    ):
        raise ValueError("all inputs must share a device")
    if not all(t.is_contiguous() for t in (q, indices, pages, attn_sink)) or any(
        t.stride()[1:] != (d, 1) for t in (window, compressed)
    ):
        raise ValueError(
            "query/metadata and each individual KV page must be contiguous"
        )
    out = torch.empty_like(q)
    if b * s:
        _sparse_sink_kernel[(s, b)](
            q,
            window,
            attn_sink,
            indices,
            out,
            s,
            h,
            d,
            window.shape[1],
            indices.shape[2],
            softmax_scale,
            max(16, triton.next_power_of_2(h)),
            64,
            compressed,
            pages,
            True,
            window.stride(0),
            compressed.stride(0),
            compressed_n,
            num_warps=8 if h >= 32 or d == 512 else 4,
            num_stages=1,
            enable_fp_fusion=True,
        )
    return out
