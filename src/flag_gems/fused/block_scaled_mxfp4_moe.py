# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe, rank-local MXFP4 routed experts with the V4.1 reduction contract.

The expert dimension is selected by a device tensor. No routing decision or
variable-sized token gather crosses back to Python during CUDA Graph capture.
The two GEMMs use the same per-32 FP8 activation and BF16 dot product as
``block_scaled_lowp_linear``; routing weights are applied before the down GEMM.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .act_quant import act_quant_triton
from .block_scaled_lowp_linear import _e2m1, _e8m0


@triton.jit
def _routed_mxfp4_mm(
    A,
    AS,
    EXPERT_IDS,
    W,
    WS,
    OUT,
    M: tl.constexpr,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FIRST_STAGE: tl.constexpr,
    LOCAL_START: tl.constexpr,
    LOCAL_COUNT: tl.constexpr,
    BN: tl.constexpr = 32,
):
    pair = tl.program_id(0)
    token = pair // TOP_K
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(EXPERT_IDS + pair) - LOCAL_START
    out_ptrs = OUT + pair * N + column
    if (expert < 0) | (expert >= LOCAL_COUNT):
        tl.store(out_ptrs, 0.0, mask=column < N)
        return

    row = token if FIRST_STAGE else pair
    k_offsets = tl.arange(0, 32)
    acc = tl.zeros((16, BN), tl.float32)
    for group in range(K // 32):
        k = group * 32 + k_offsets
        # The reference GEMM uses BM=16 even for one routed row. Other rows
        # are zero and only row zero is written back.
        a = tl.load(A + row * K + k, mask=k < K, other=0.0).to(tl.bfloat16)
        a = tl.where(tl.arange(0, 16)[:, None] == 0, a[None, :], 0.0).to(
            tl.bfloat16
        )
        packed = tl.load(
            W + expert * N * (K // 2) + column[None, :] * (K // 2) + k[:, None] // 2,
            mask=column[None, :] < N,
            other=0,
        ).to(tl.uint8)
        codes = (packed >> ((k[:, None] & 1) * 4)) & 15
        w = _e2m1(codes).to(tl.bfloat16)
        a_scale = tl.load(AS + row * (K // 32) + group)
        w_code = tl.load(
            WS + expert * N * (K // 32) + column * (K // 32) + group,
            mask=column < N,
            other=127,
        )
        product = tl.dot(a, w)
        acc = acc + product * a_scale * _e8m0(w_code)[None, :]
    tl.store(out_ptrs, tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, acc, 0.0), 0), mask=column < N)


def block_scaled_mxfp4_moe(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
    *,
    local_start: int,
    swiglu_limit: float,
) -> torch.Tensor:
    """Return rank-local FP32 routed output for ``x`` of shape ``[M, H]``.

    Packed weights have shapes ``[E, 2I, H/2]`` and ``[E, H, I/2]``;
    scale bytes have the corresponding K/32 last dimensions. The caller adds
    the shared expert and performs the expert-parallel reduction.
    """
    if x.ndim != 2 or expert_ids.ndim != 2 or expert_ids.shape != routing_weights.shape:
        raise ValueError("expected x[M,H] and matching expert_ids/weights[M,top_k]")
    m, hidden = x.shape
    top_k = expert_ids.shape[1]
    local_count, twice_intermediate, packed_hidden = gate_up.shape
    intermediate = twice_intermediate // 2
    if (
        expert_ids.shape[0] != m
        or twice_intermediate != 2 * intermediate
        or packed_hidden * 2 != hidden
        or down.shape != (local_count, hidden, intermediate // 2)
        or gate_up_scale.shape != (local_count, 2 * intermediate, hidden // 32)
        or down_scale.shape != (local_count, hidden, intermediate // 32)
        or hidden % 32 or intermediate % 32
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or gate_up.dtype != torch.uint8
        or down.dtype != torch.uint8
        or gate_up_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
        or down_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
    ):
        raise ValueError("invalid contiguous MXFP4 expert layout or activation")
    if not all(t.is_contiguous() for t in (expert_ids, routing_weights, gate_up, gate_up_scale, down, down_scale)):
        raise ValueError("routing and expert tensors must be contiguous")
    if m == 0:
        return torch.empty_like(x, dtype=torch.float32)

    # Sorting reproduces the reference's ascending expert-ID accumulation.
    expert_ids, order = expert_ids.sort(dim=-1)
    routing_weights = routing_weights.gather(1, order)
    quantized, scales = act_quant_triton(x, 32, "ue8m0")
    first = torch.empty((m, top_k, 2 * intermediate), device=x.device, dtype=x.dtype)
    _routed_mxfp4_mm[(m * top_k, triton.cdiv(2 * intermediate, 32))](
        quantized, scales, expert_ids, gate_up, gate_up_scale.view(torch.uint8), first,
        m, top_k, 2 * intermediate, hidden, True, local_start, local_count,
        num_warps=4, enable_fp_fusion=False,
    )
    gate, up = first.float().split(intermediate, dim=-1)
    if swiglu_limit > 0:
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    activated = (F.silu(gate) * up * routing_weights.unsqueeze(-1)).to(x.dtype)
    activated = activated.contiguous().view(m * top_k, intermediate)
    quantized, scales = act_quant_triton(activated, 32, "ue8m0")
    second = torch.empty((m, top_k, hidden), device=x.device, dtype=x.dtype)
    _routed_mxfp4_mm[(m * top_k, triton.cdiv(hidden, 32))](
        quantized, scales, expert_ids, down, down_scale.view(torch.uint8), second,
        m, top_k, hidden, intermediate, False, local_start, local_count,
        num_warps=4, enable_fp_fusion=False,
    )
    # Match the reference's FP32 accumulation in ascending expert order.
    result = second[:, 0].float()
    for slot in range(1, top_k):
        result = result + second[:, slot].float()
    return result
