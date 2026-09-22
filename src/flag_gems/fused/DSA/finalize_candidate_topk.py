# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact/dense top-k columns to physical and logical cache slots.

Ported from CherryLemon/vllm a9e3d217cce075c77a8041d14dd822307953735e,
model_executor/kernels/attention/dsa/candidate_blocks.py. Pure Triton;
no serving-framework dependency. NaN/-inf/invalid columns map to -1,
+inf remains selectable, and compact candidates are mapped before applying
logical visibility. Input selection order is preserved, not sorted here.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _finalize_candidate_topk_kernel(
    selected_ptr,
    scores_ptr,
    score_lens_ptr,
    block_table_ptr,
    candidate_blocks_ptr,
    page_indices_ptr,
    raw_indices_ptr,
    stride_sel,
    stride_score,
    stride_bt,
    stride_cb,
    stride_out,
    stride_raw,
    block_size,
    TOPK: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    """Map selected candidate-token columns to vLLM compressed positions.

    ``selected`` are TopK columns into the candidate-token logits (one row per
    query).  Candidate mode resolves column ``c`` to request-local position
    ``candidate_blocks[row, c // CBS] * CBS + c % CBS``; a selected column whose
    logit is ``-inf`` (padding, NaN or beyond the context) is dropped.  The
    physical slot of logical position ``L`` is
    ``block_table[row, L // block_size] * block_size + L % block_size``
    (RATIO == 1: no req_to_token indirection).  ``raw_indices`` receives the
    request-local positions; ``page_indices`` the physical slots.  Both are -1
    padded, with the same column order as ``selected``. Compact candidate ids
    may be unsorted; this mapping does not sort logical or physical outputs.
    """
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, TOPK)
    selected = tl.load(selected_ptr + row * stride_sel + offs).to(tl.int64)
    length = tl.load(score_lens_ptr + row).to(tl.int64)
    valid = (selected >= 0) & (selected < SOURCE_WIDTH)
    score = tl.load(
        scores_ptr + row * stride_score + tl.maximum(selected, 0),
        mask=valid,
        other=-float("inf"),
    )
    valid = valid & (score > -float("inf"))
    if USE_CANDIDATES:
        block_col = selected // CANDIDATE_BLOCK_SIZE
        within = selected % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + row * stride_cb + tl.maximum(block_col, 0),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (block >= 0)
        logical = block * CANDIDATE_BLOCK_SIZE + within
    else:
        logical = selected
    valid = valid & (logical < length)
    safe_logical = tl.where(valid, logical, 0)
    page = tl.load(
        block_table_ptr + row * stride_bt + safe_logical // block_size,
        mask=valid,
        other=0,
    ).to(tl.int64)
    slot = page * block_size + (safe_logical % block_size)
    tl.store(page_indices_ptr + row * stride_out + offs, tl.where(valid, slot, -1))
    if HAS_RAW:
        tl.store(
            raw_indices_ptr + row * stride_raw + offs, tl.where(valid, logical, -1)
        )


def finalize_candidate_topk(
    selected: torch.Tensor,
    scores: torch.Tensor,
    score_lens: torch.Tensor,
    block_table: torch.Tensor,
    page_indices: torch.Tensor,
    *,
    block_size: int,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 1,
    raw_indices: torch.Tensor | None = None,
) -> None:
    """Finalize a compact candidate TopK into vLLM indexer slots.

    Args:
        selected: ``[rows, k]`` int32 TopK columns over the candidate-token
            logits, ``-1`` padded.  ``k`` must equal ``page_indices.shape[1]``
            (the kernel's TOPK is a constexpr) and be a power of two.
        scores: ``[rows, SOURCE_WIDTH]`` fp32 candidate-token logits; used to
            reject selections whose score is NaN or -inf (+inf is valid).
        score_lens: ``[rows]`` int32 visible (compressed) length per row.
        block_table: ``[rows, P]`` int32 page table (``stride(-1) == 1``).
        page_indices: ``[rows, k]`` int32 output of physical cache slots.
        block_size: Indexer cache tokens per page.
        candidate_blocks: ``[rows, K]`` int32 request-local candidate blocks
            (-1 padded).  None means ``selected`` is already a local position.
        candidate_block_size: Positions per candidate block.
        raw_indices: Optional ``[rows, k]`` int32 output of request-local
            compressed positions.  Must not alias ``selected``.

    """
    assert selected.dtype == torch.int32 and selected.stride(-1) == 1
    assert page_indices.dtype == torch.int32 and page_indices.stride(-1) == 1
    assert scores.dtype == torch.float32 and scores.stride(-1) == 1
    assert block_size > 0 and candidate_block_size > 0
    assert selected.ndim == page_indices.ndim == scores.ndim == block_table.ndim == 2
    k = page_indices.shape[1]
    # The report flags this invariant as a risk: the Triton kernel's TOPK is a
    # constexpr and the store covers exactly page_indices.shape[1] columns.
    assert selected.shape[1] == k, (
        "finalize_candidate_topk requires k == page_indices.shape[1], got "
        f"{selected.shape[1]} != {k}"
    )
    assert triton.next_power_of_2(k) == k, f"k must be a power of two, got {k}"
    rows = selected.shape[0]
    assert page_indices.shape[0] == scores.shape[0] == rows
    if rows == 0:
        return
    use_candidates = candidate_blocks is not None
    if use_candidates:
        assert (
            candidate_blocks.dtype == torch.int32 and candidate_blocks.stride(-1) == 1
        )
        assert candidate_blocks.shape[0] == rows
        assert candidate_blocks.shape[1] * candidate_block_size >= scores.shape[1]
    assert score_lens.dtype == torch.int32 and score_lens.shape == (rows,)
    assert block_table.shape[0] == rows and block_table.stride(-1) == 1
    if raw_indices is not None:
        assert raw_indices.dtype == torch.int32 and raw_indices.stride(-1) == 1
        assert raw_indices.shape[0] == rows and raw_indices.shape[1] == k
    tensors = [scores, score_lens, block_table, page_indices]
    if candidate_blocks is not None:
        tensors.append(candidate_blocks)
    if raw_indices is not None:
        tensors.append(raw_indices)
    assert all(t.device == selected.device for t in tensors)
    for out in (page_indices, raw_indices):
        if out is not None:
            assert (
                out.untyped_storage().data_ptr()
                != selected.untyped_storage().data_ptr()
            ), "selected must not alias output"
    _finalize_candidate_topk_kernel[(rows,)](
        selected,
        scores,
        score_lens,
        block_table,
        candidate_blocks if use_candidates else selected,
        page_indices,
        raw_indices if raw_indices is not None else page_indices,
        selected.stride(0),
        scores.stride(0),
        block_table.stride(0),
        candidate_blocks.stride(0) if use_candidates else 0,
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        block_size,
        TOPK=k,
        SOURCE_WIDTH=scores.shape[1],
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        USE_CANDIDATES=use_candidates,
        HAS_RAW=raw_indices is not None,
        num_warps=8,
    )
