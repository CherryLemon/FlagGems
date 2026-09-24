# SPDX-License-Identifier: Apache-2.0
"""Batch independent decode reductions without changing their association.

The reduction geometry matches contiguous FP32 ATen Reduce.cuh (CUDA): four
independent vector accumulators, descending block-x reduction, then block-y.
The number of rows in one reference request determines that geometry; the
number of requests only determines our grid. This avoids one launch per
request while retaining rounding at the original reduction boundary.
"""

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


def _reference_geometry(columns, reference_rows):
    dim0 = min(512, 1 << (columns // 4).bit_length() - 1)
    dim1 = min(512, 1 << reference_rows.bit_length() - 1)
    height = min(dim1, 512 // min(dim0, 32))
    width = min(dim0, 512 // height)
    split_y = triton.cdiv(columns, width) >= min(height * 16, 256)
    return width, height if split_y else 1


@libentry()
@triton.jit
def _decode_row_reduce(
    X,
    Out,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    HEIGHT: tl.constexpr,
    MEAN: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, HEIGHT)[:, None] * WIDTH + tl.arange(0, WIDTH)[None, :]
    v0 = tl.full((HEIGHT, WIDTH), 0, tl.float32)
    v1 = tl.full((HEIGHT, WIDTH), 0, tl.float32)
    v2 = tl.full((HEIGHT, WIDTH), 0, tl.float32)
    v3 = tl.full((HEIGHT, WIDTH), 0, tl.float32)
    for start in range(tl.cdiv(K, WIDTH * HEIGHT * 4)):
        offset = (lane + start * WIDTH * HEIGHT) * 4
        valid = offset < K
        v0 += tl.load(X + row * K + offset, valid, 0)
        v1 += tl.load(X + row * K + offset + 1, valid, 0)
        v2 += tl.load(X + row * K + offset + 2, valid, 0)
        v3 += tl.load(X + row * K + offset + 3, valid, 0)
    value = ((v0 + v1) + v2) + v3
    # A plain tl.sum can reduce within warps before combining warps. ATen
    # combines the large block-x halves first, then the 32-lane warp tree.
    for shift in tl.static_range(WIDTH.bit_length() - 2, -1, -1):
        partner = tl.broadcast_to(
            (tl.arange(0, WIDTH) ^ (1 << shift))[None, :], (HEIGHT, WIDTH)
        )
        value = value + tl.gather(value, partner, 1)
    value = tl.sum(tl.where(tl.arange(0, WIDTH)[None, :] == 0, value, 0), 1)
    for shift in tl.static_range(HEIGHT.bit_length() - 2, -1, -1):
        value = value + tl.gather(value, tl.arange(0, HEIGHT) ^ (1 << shift), 0)
    value = tl.sum(tl.where(tl.arange(0, HEIGHT) == 0, value, 0), 0)
    if MEAN:
        value *= 1.0 / K
    tl.store(Out + row, value)


def decode_row_reduce(x, *, reference_rows=1, mean=True, keepdim=False):
    """Reduce the last dimension using each request's FP32 CUDA geometry.

    Explicit supported envelope: contiguous, aligned FP32 rows, K divisible
    by four in [128, 32768]. Reference rows is a static shape, never GPU data.
    No inter-CTA reduction or workspace is needed in this envelope.
    """
    if (
        x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.ndim < 1
        or x.shape[-1] < 128
        or x.shape[-1] > 32768
        or x.shape[-1] % 4
        or x.storage_offset() % 4
        or reference_rows < 1
    ):
        raise ValueError(
            "expected aligned contiguous FP32 rows, 128 <= K <= 32768, K % 4 == 0"
        )
    rows = x.numel() // x.shape[-1]
    if rows % reference_rows:
        raise ValueError("rows must retain complete reference request groups")
    width, height = _reference_geometry(x.shape[-1], reference_rows)
    out = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
    if rows:
        _decode_row_reduce[(rows,)](
            x,
            out,
            x.shape[-1],
            width,
            height,
            mean,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out.unsqueeze(-1) if keepdim else out
