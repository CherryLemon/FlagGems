# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Device-indexed request state operations for graph-captured V4.1 decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_rows(
    Pool,
    Values,
    Pages,
    Positions,
    Active,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, batch = tl.program_id(0), tl.program_id(1)
    page = tl.load(Pages + batch)
    position = tl.load(Positions + batch * S + token)
    active = tl.load(Active + batch * S + token)
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Values + (batch * S + token) * D + cols, cols < D, 0)
    tl.store(
        Pool + page * STRIDE + position * D + cols,
        value,
        (cols < D) & active & (position >= 0) & (position < N),
    )


def write_request_rows(pool, values, pages, positions, active):
    """Write values[B,S,D] to unique (page,position) rows if active[B,S].

    Pools are [P,N,D] with a possibly padded page stride. Callers must supply
    in-range page IDs and avoid duplicate active destinations. Inactive lanes
    never write, including graph padding and rejected speculative suffixes.
    """
    if pool.ndim != 3 or values.ndim != 3:
        raise ValueError("expected pool[P,N,D] and values[B,S,D]")
    b, s, d = values.shape
    if (
        pool.shape[2] != d
        or pool.dtype != values.dtype
        or pool.stride()[1:] != (d, 1)
        or pages.shape != (b,)
        or pages.dtype != torch.int64
        or positions.shape != (b, s)
        or positions.dtype != torch.int64
        or active.shape != (b, s)
        or active.dtype != torch.bool
        or not all(t.is_contiguous() for t in (values, pages, positions, active))
        or any(t.device != pool.device for t in (values, pages, positions, active))
    ):
        raise ValueError("invalid request-row layout or metadata")
    if b * s:
        _write_rows[(s, b, triton.cdiv(d, 256))](
            pool,
            values,
            pages,
            positions,
            active,
            s,
            d,
            pool.shape[1],
            pool.stride(0),
            256,
        )


@triton.jit
def _index_scores(
    Q,
    K,
    W,
    Pages,
    Lengths,
    Out,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    block, token, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    page = tl.load(Pages + batch)
    length = tl.load(Lengths + batch * S + token)
    heads, dims = tl.arange(0, BH), tl.arange(0, D)
    cols = block * BN + tl.arange(0, BN)
    q = tl.load(
        Q + ((batch * S + token) * H + heads[:, None]) * D + dims[None, :],
        heads[:, None] < H,
        0,
    )
    k = tl.load(
        K + page * STRIDE + cols[None, :] * D + dims[:, None],
        (cols[None, :] < N) & (cols[None, :] < length),
        0,
    )
    # Match einsum's BF16 output followed by BF16 relu * head weights.
    dots = tl.dot(q, k).to(q.dtype).to(tl.float32)
    weights = tl.load(W + (batch * S + token) * H + heads, heads < H, 0)
    weighted = (tl.maximum(dots, 0.0) * weights[:, None]).to(q.dtype)
    score = tl.sum(weighted.to(tl.float32), 0).to(q.dtype)
    score = tl.where(cols < length, score, -float("inf"))
    tl.store(Out + (batch * S + token) * N + cols, score, cols < N)


def paged_index_scores(q, keys, weights, pages, lengths):
    """BF16 per-head dot/ReLU/weight/sum without materializing gathered K.

    q[B,S,H,D], keys[P,N,D], weights[B,S,H], visible lengths[B,S].
    The fixed output capacity is N; invisible entries are -inf. All request
    and position choices come from device metadata, not capture-time scalars.
    V4.1 supplies Q/K already rounded to E2M1 with per-32 E8M0 scales and
    decoded to BF16. Bitwise reference checks use that quantization contract;
    arbitrary BF16 dots may differ from cuBLAS at an FP32 rounding tie.
    """
    if q.ndim != 4 or keys.ndim != 3:
        raise ValueError("expected q[B,S,H,D] and keys[P,N,D]")
    b, s, h, d = q.shape
    if (
        keys.shape[2] != d
        or keys.stride()[1:] != (d, 1)
        or q.dtype != torch.bfloat16
        or keys.dtype != q.dtype
        or weights.dtype != q.dtype
        or weights.shape != (b, s, h)
        or pages.dtype != torch.int64
        or pages.shape != (b,)
        or lengths.dtype != torch.int64
        or lengths.shape != (b, s)
        or not 0 < h <= 64
        or d not in (32, 64, 128, 256)
        or not all(t.is_contiguous() for t in (q, weights, pages, lengths))
        or any(t.device != q.device for t in (keys, weights, pages, lengths))
    ):
        raise ValueError("invalid paged index scoring inputs")
    out = torch.empty((b, s, keys.shape[1]), dtype=q.dtype, device=q.device)
    if b * s and keys.shape[1]:
        _index_scores[(triton.cdiv(keys.shape[1], 64), s, b)](
            q,
            keys,
            weights,
            pages,
            lengths,
            out,
            s,
            h,
            d,
            keys.shape[1],
            keys.stride(0),
            max(16, triton.next_power_of_2(h)),
            64,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
