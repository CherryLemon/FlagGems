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

# Model-shape H100 Graph sweeps: 128 beats 32/64/256 without changing the
# per-32 reduction. This is a fixed capture-safe configuration, not autotuning.
_GROUP_BN = 128


def _grouped_moe_block_m(m):
    # Small decode batches favor less padding; batched verification benefits
    # from wider expert tiles on the strongly correlated serving routes.
    return 16 if m < 16 else 32 if m < 256 else 64


@triton.jit
def _group_expert_pairs(
    IDS,
    PAIRS,
    COUNTS,
    P: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    BP: tl.constexpr,
):
    expert = tl.program_id(0)
    pair = tl.arange(0, BP)
    selected = (pair < P) & (tl.load(IDS + pair, pair < P, other=-1) == expert + START)
    ordinal = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(PAIRS + expert * P + ordinal, pair, selected)
    tl.store(COUNTS + expert, tl.sum(selected.to(tl.int32)))


@triton.jit
def _map_expert_tiles(
    COUNTS, EXPERTS, OFFSETS, E: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr
):
    tile = tl.program_id(0)
    experts = tl.arange(0, BE)
    counts = tl.load(COUNTS + experts, experts < E, other=0)
    ends = tl.cumsum(tl.cdiv(counts, BM))
    expert = tl.min(tl.where((experts < E) & (ends > tile), experts, E))
    begin = tl.sum(tl.where(experts == expert - 1, ends, 0))
    tl.store(EXPERTS + tile, expert)
    tl.store(OFFSETS + tile, (tile - begin) * BM)


@triton.jit
def _grouped_routed_mxfp4_mm(
    A,
    AS,
    PAIRS,
    COUNTS,
    EXPERTS,
    OFFSETS,
    W,
    WS,
    OUT,
    P: tl.constexpr,
    TOP_K: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FIRST_STAGE: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
):
    tile = tl.program_id(0)
    expert = tl.load(EXPERTS + tile)
    if expert >= E:
        return
    rows = tl.load(OFFSETS + tile) + tl.arange(0, BM)
    valid = rows < tl.load(COUNTS + expert)
    pairs = tl.load(PAIRS + expert * P + rows, valid, other=0)
    a_rows = pairs // TOP_K if FIRST_STAGE else pairs
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    # Keep the reference's per-32 product, scale and accumulation order.
    # Grouping changes independent rows, never the K reduction contract.
    for group in range(K // 32):
        k = group * 32 + offsets
        a = tl.load(A + a_rows[:, None] * K + k[None, :], valid[:, None], other=0.0).to(
            tl.bfloat16
        )
        packed = tl.load(
            W + expert * N * (K // 2) + columns[None, :] * (K // 2) + k[:, None] // 2,
            columns[None, :] < N,
            other=0,
        ).to(tl.uint8)
        codes = (packed >> ((k[:, None] & 1) * 4)) & 15
        w = _e2m1(codes).to(tl.bfloat16)
        a_scale = tl.load(AS + a_rows * (K // 32) + group, valid, other=0)
        w_code = tl.load(
            WS + expert * N * (K // 32) + columns * (K // 32) + group,
            columns < N,
            other=127,
        )
        acc = acc + tl.dot(a, w) * a_scale[:, None] * _e8m0(w_code)[None, :]
    tl.store(
        OUT + pairs[:, None] * N + columns[None, :],
        acc,
        valid[:, None] & (columns[None, :] < N),
    )


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
        a = tl.where(tl.arange(0, 16)[:, None] == 0, a[None, :], 0.0).to(tl.bfloat16)
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
    tl.store(
        out_ptrs,
        tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, acc, 0.0), 0),
        mask=column < N,
    )


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
    implementation: str = "grouped",
) -> torch.Tensor:
    """Return rank-local FP32 routed output for ``x`` of shape ``[M, H]``.

    Packed weights have shapes ``[E, 2I, H/2]`` and ``[E, H, I/2]``;
    scale bytes have the corresponding K/32 last dimensions. The caller adds
    the shared expert and performs the expert-parallel reduction.
    """
    if implementation not in ("grouped", "reference"):
        raise ValueError("implementation must be grouped or reference")
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
        or hidden % 32
        or intermediate % 32
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or gate_up.dtype != torch.uint8
        or down.dtype != torch.uint8
        or gate_up_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
        or down_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
    ):
        raise ValueError("invalid contiguous MXFP4 expert layout or activation")
    if not all(
        t.is_contiguous()
        for t in (expert_ids, routing_weights, gate_up, gate_up_scale, down, down_scale)
    ):
        raise ValueError("routing and expert tensors must be contiguous")
    if m == 0:
        return torch.empty_like(x, dtype=torch.float32)

    # Sorting reproduces the reference's ascending expert-ID accumulation.
    expert_ids, order = expert_ids.sort(dim=-1)
    routing_weights = routing_weights.gather(1, order)
    if implementation == "grouped":
        pair_count, bm = m * top_k, _grouped_moe_block_m(m)
        # Sum ceil(count_e/BM) <= ceil(P/BM) + E-1. Empty experts and
        # highly skewed routes fit this fixed, graph-safe upper bound.
        max_tiles = triton.cdiv(pair_count, bm) + local_count - 1
        pairs = torch.empty(
            (local_count, pair_count), device=x.device, dtype=torch.int32
        )
        counts = torch.empty(local_count, device=x.device, dtype=torch.int32)
        tile_experts = torch.empty(max_tiles, device=x.device, dtype=torch.int32)
        tile_offsets = torch.empty_like(tile_experts)
        _group_expert_pairs[(local_count,)](
            expert_ids,
            pairs,
            counts,
            pair_count,
            local_start,
            bm,
            triton.next_power_of_2(pair_count),
        )
        _map_expert_tiles[(max_tiles,)](
            counts,
            tile_experts,
            tile_offsets,
            local_count,
            bm,
            triton.next_power_of_2(local_count),
        )

    def matmul(quantized, scales, weight, weight_scale, n, k, first_stage):
        out = torch.zeros((m, top_k, n), device=x.device, dtype=x.dtype)
        if implementation == "grouped":
            _grouped_routed_mxfp4_mm[(max_tiles, triton.cdiv(n, _GROUP_BN))](
                quantized,
                scales,
                pairs,
                counts,
                tile_experts,
                tile_offsets,
                weight,
                weight_scale.view(torch.uint8),
                out,
                pair_count,
                top_k,
                local_count,
                n,
                k,
                first_stage,
                BM=bm,
                BN=_GROUP_BN,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _routed_mxfp4_mm[(m * top_k, triton.cdiv(n, 32))](
                quantized,
                scales,
                expert_ids,
                weight,
                weight_scale.view(torch.uint8),
                out,
                m,
                top_k,
                n,
                k,
                first_stage,
                local_start,
                local_count,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out

    quantized, scales = act_quant_triton(x, 32, "ue8m0")
    first = matmul(
        quantized, scales, gate_up, gate_up_scale, 2 * intermediate, hidden, True
    )
    gate, up = first.float().split(intermediate, dim=-1)
    if swiglu_limit > 0:
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    activated = (F.silu(gate) * up * routing_weights.unsqueeze(-1)).to(x.dtype)
    activated = activated.contiguous().view(m * top_k, intermediate)
    quantized, scales = act_quant_triton(activated, 32, "ue8m0")
    second = matmul(quantized, scales, down, down_scale, hidden, intermediate, False)
    # Match the reference's FP32 accumulation in ascending expert order.
    result = second[:, 0].float()
    for slot in range(1, top_k):
        result = result + second[:, slot].float()
    return result
