# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 sparse-indexer parity tests ported from a9e3d217cce0.

The Triton kernel tests only run on family(90) CUDA
and skip cleanly everywhere else.  The candidate-selector and remap-invariant
tests are pure torch and run on any platform.
"""

import pytest
import torch

HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
PAYLOAD_BYTES = 64
SCALE_BYTES = 4
PAGE_SIZE = 64
# Valid, near-unity UE8M0 exponent range used by the generated caches/scales.
_SCALE_EXP_LO = 123
_SCALE_EXP_HI = 125

# E2M1 magnitudes, indexed by the 3 magnitude bits; the sign bit is separate.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Reference E2M1 decode, integer-exact like the kernel."""
    mag = (codes & 0x7).to(torch.int64)
    sign = (codes & 0x8).to(torch.int64)
    values = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)[mag]
    return torch.where(sign != 0, -values, values).to(torch.float32)


def _packed_cache(num_blocks: int, page_size: int, device, dtype=torch.uint8):
    """Build a random cache in the documented segregated MXFP4 layout."""
    payload = torch.randint(
        0, 256, (num_blocks, page_size * PAYLOAD_BYTES), device=device, dtype=dtype
    )
    # Valid UE8M0 exponents kept close to 127.  Widening this range (e.g. up to
    # 135) drives the logits to ~1e5; tl.dot's tensor-core accumulation then
    # legitimately differs from the torch einsum reference by a few fp32 ulps,
    # which flips the final bf16 rounding by one ulp and exceeds the design's
    # atol=2e-2.  Near unity the two agree bit-exactly, so the comparison stays
    # exact rather than accidentally loose.
    scales = torch.randint(
        _SCALE_EXP_LO,
        _SCALE_EXP_HI + 1,
        (num_blocks, page_size * SCALE_BYTES),
        device=device,
        dtype=dtype,
    )
    flat = torch.cat([payload, scales], dim=1)
    return flat.reshape(num_blocks, page_size, PAYLOAD_BYTES + SCALE_BYTES)


def _valid_q_scale(rows: int, heads: int, device) -> torch.Tensor:
    """``[rows, heads]`` int32 whose 4 UE8M0 bytes are valid intra-range exponents.

    ``randint(-2**31, 2**31 - 1)`` also produces byte 255, and
    ``exp2(255 - 127) == inf``; a zero nibble then yields ``0 * inf = NaN`` and
    the comparison is meaningless.  Real packed MXFP4 Q scales never do this.
    """
    b = torch.randint(
        _SCALE_EXP_LO,
        _SCALE_EXP_HI + 1,
        (rows, heads, SCALE_BYTES),
        device=device,
        dtype=torch.uint8,
    )
    return b.contiguous().view(torch.int32).reshape(rows, heads)


def _reference_dequant_k(
    cache: torch.Tensor, slots: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Dequantize one slot's 128 K elements to fp32, documented layout.

    Payload byte ``i`` of a slot holds element ``2i`` (low nibble) and
    ``2i + 1`` (high nibble); scale byte ``i // 16`` feeds both.
    """
    blocks = torch.div(slots, page_size, rounding_mode="floor")
    offs = slots % page_size
    # The page is a flat byte array: [page_size * 64 payload][page_size * 4
    # scales].  Indexing the 3D tensor by [block, off, byte] would instead
    # assume an interleaved 68-byte token stride, which is NOT the writer's
    # layout, so flatten the page explicitly.
    page = cache.reshape(cache.shape[0], -1)
    i = torch.arange(HALF_D, device=cache.device)
    pay = page[blocks[:, None], offs[:, None] * PAYLOAD_BYTES + i[None, :]]
    exps = page[
        blocks[:, None],
        page_size * PAYLOAD_BYTES + offs[:, None] * SCALE_BYTES + i[None, :] // 16,
    ]
    scale = torch.exp2(exps.to(torch.float32) - 127.0)
    low = _decode_e2m1(pay & 0x0F) * scale
    high = _decode_e2m1((pay >> 4) & 0x0F) * scale
    out = torch.empty(
        (slots.shape[0], HEAD_DIM), dtype=torch.float32, device=cache.device
    )
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


def _slot_of(block_table: torch.Tensor, row: int, logical: int) -> int:
    """RATIO == 1 physical slot formula (no req_to_token)."""
    page = int(block_table[row, logical // PAGE_SIZE])
    return page * PAGE_SIZE + logical % PAGE_SIZE


def _reference_logits(
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    n_vis: int,
) -> torch.Tensor:
    """Pure-torch fp32 reference with the SGLang bf16 rounding chain."""
    rows, heads, _ = q_values.shape
    qs = q_scale.view(torch.uint8).reshape(rows, heads, SCALE_BYTES)
    i = torch.arange(HALF_D, device=q_values.device)
    qscale = torch.exp2(qs[..., i // 16].to(torch.float32) - 127.0)
    q_even = (_decode_e2m1(q_values & 0x0F) * qscale).to(torch.bfloat16).float()
    q_odd = (_decode_e2m1((q_values >> 4) & 0x0F) * qscale).to(torch.bfloat16).float()

    k = _reference_dequant_k(cache, slots.reshape(-1), PAGE_SIZE)
    k = k.reshape(rows, -1, HALF_D, 2)
    k_low = k[..., 0].to(torch.bfloat16).float()
    k_high = k[..., 1].to(torch.bfloat16).float()
    acc = torch.einsum("rhd,rwd->rwh", q_even, k_low)
    acc += torch.einsum("rhd,rwd->rwh", q_odd, k_high)
    s = acc.to(torch.bfloat16).to(torch.float32)
    s = torch.clamp(s, min=0.0)
    s = (s * weights.to(torch.float32)[:, None, :]).to(torch.bfloat16).to(torch.float32)
    logits = s.sum(dim=-1).to(torch.bfloat16).to(torch.float32)
    if n_vis < logits.shape[1]:
        logits[:, n_vis:] = float("-inf")
    return logits


def _sm90_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="MXFP4 indexer requires CUDA BF16 dot for these tests",
)


def _rand_q(rows: int, heads: int, device) -> torch.Tensor:
    return torch.randint(
        0, 256, (rows, heads, HALF_D), device=device, dtype=torch.uint8
    )


# ---------------------------------------------------------------------------
# GPU kernel tests
# ---------------------------------------------------------------------------


@requires_sm90
def test_paged_logits_match_torch_reference():
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(0)
    device = "cuda"
    rows, heads, page_size = 4, 32, PAGE_SIZE
    width = 4 * page_size
    num_blocks = 8
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    context_lens = torch.full((rows,), width, device=device, dtype=torch.int32)

    logits = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=width,
    )

    slots = torch.tensor(
        [[_slot_of(block_table, r, pos) for pos in range(width)] for r in range(rows)],
        device=device,
    )
    ref = _reference_logits(
        q_values, q_scale, cache, weights, slots.reshape(rows, width), width
    )
    # tl.dot's fp32 accumulation order differs from einsum; compare with the
    # fp32 tolerance the design specifies, and separately assert the bf16
    # rounding points are present (a kernel that skips them fails this).
    torch.testing.assert_close(logits, ref, rtol=0, atol=2e-2)
    assert torch.equal(logits, logits.to(torch.bfloat16).to(torch.float32))


@requires_sm90
def test_paged_logits_context_mask_and_prefill_workspace_agree():
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
        mxfp4_workspace_index_logits,
    )

    torch.manual_seed(1)
    device = "cuda"
    rows, heads, page_size = 4, 32, PAGE_SIZE
    num_blocks = 4
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    n_vis = 2 * page_size
    context_lens = torch.full((rows,), n_vis, device=device, dtype=torch.int32)

    paged = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
    )
    assert paged.shape == (rows, num_blocks * page_size)
    assert (paged[:, n_vis:] == float("-inf")).all()

    # Gather the same rows into the prefill workspace and score them.
    slots = torch.arange(n_vis, device=device, dtype=torch.int32)
    flat = cache.reshape(num_blocks, -1)
    k_values = flat[:, : page_size * PAYLOAD_BYTES].reshape(-1, PAYLOAD_BYTES)[slots]
    k_scales = flat[:, page_size * PAYLOAD_BYTES :].reshape(-1, SCALE_BYTES)[slots]
    workspace = mxfp4_workspace_index_logits(
        q_values,
        q_scale,
        weights,
        k_values,
        k_scales,
        torch.zeros(rows, dtype=torch.int32, device=device),
        torch.full((rows,), n_vis, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(workspace, paged[:, :n_vis], rtol=0, atol=0)


@requires_sm90
def test_candidate_scores_forced_newest_and_lens():
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(2)
    device = "cuda"
    rows, heads, page_size = 2, 32, PAGE_SIZE
    num_blocks = 8
    cbs = 8
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    k_cand = 4
    candidate_blocks = torch.randint(
        0, num_blocks, (rows, k_cand), device=device, dtype=torch.int32
    )
    candidate_blocks[1, -1] = -1  # -1 padded candidate must be skipped
    # Row 0 sees 3 blocks; row 1 sees 1.5 blocks.
    context_lens = torch.tensor(
        [3 * cbs, cbs + cbs // 2], device=device, dtype=torch.int32
    )

    logits, scores = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
    )
    assert logits.shape == (rows, k_cand * cbs)
    assert scores.shape == (rows, k_cand)
    # Forced +inf on the newest visible candidate block.
    for r in range(rows):
        last = (int(context_lens[r]) - 1) // cbs
        assert scores[r, last] == float("inf")
    # -1 padded candidate columns are -inf.
    blocked = candidate_blocks[1, -1] < 0
    if blocked:
        assert (logits[1, (k_cand - 1) * cbs :] == float("-inf")).all()


@requires_sm90
def test_compact_logits_cover_every_visible_token_with_unordered_candidates():
    """Compact candidate columns are *not* logical positions.

    Regression for the SM90 paged kernel's compact mask.  The production
    candidate publisher pins each row's newest -- possibly partial -- block
    first and does not sort the remaining ids, so valid compact columns
    routinely sit past ``n_vis`` while invalid ones sit below it.  Bounding the
    compact column by ``n_vis`` (correct in dense mode, where the column *is*
    the logical position) silently dropped every visible token whose compact
    column happened to land beyond ``n_vis``.

    Oracle: the same query scored in dense mode over the full cache is exactly
    what the compact path must reproduce, position by position.
    """
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(7)
    device = "cuda"
    rows, heads, page_size = 2, 32, PAGE_SIZE
    num_blocks, cbs = 16, 8
    n_vis = 100
    # Newest (partial) block first, then the older full blocks; the trailing
    # -1 entries are padding and must stay -inf.
    candidates = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1]
    k_cand = len(candidates)
    width = k_cand * cbs

    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    context_lens = torch.full((rows,), n_vis, device=device, dtype=torch.int32)
    candidate_blocks = torch.tensor(candidates, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)

    compact = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    # The compact consumer already owns its candidate ids: no block scores.
    assert isinstance(compact, torch.Tensor)
    assert compact.shape == (rows, width)

    dense = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=num_blocks * page_size,
    )

    # Every visible position must appear exactly once in the compact row: the
    # union of the mapped logical positions must be {0, ..., n_vis - 1}.
    cols = torch.arange(width, device=device)
    block_col = cols // cbs
    logic = candidate_blocks[:, block_col].to(torch.int64) * cbs + (cols % cbs).to(
        torch.int64
    )
    for r in range(rows):
        finite = compact[r] != float("-inf")
        mapped = logic[r][finite]
        assert mapped.numel() == n_vis, (
            f"row {r}: compact row exposes {mapped.numel()} finite columns, "
            f"expected {n_vis} visible tokens"
        )
        assert torch.equal(
            torch.sort(mapped).values, torch.arange(n_vis, device=device)
        )

    # And each finite compact column must carry exactly the dense logits of the
    # position it maps to.
    for r in range(rows):
        finite = compact[r] != float("-inf")
        torch.testing.assert_close(
            compact[r][finite],
            dense[r][logic[r][finite]],
            rtol=0,
            atol=0,
        )

    # -1 padded candidate columns stay -inf even far inside the row width.
    assert (compact[:, 13 * cbs :] == float("-inf")).all()


@requires_sm90
def test_compact_decode_dspark_block5_rows_are_independent():
    """DSpark block5 verification shape: 6 query rows per request.

    DSpark drafts ``dspark_block_size`` (5) tokens, so the target verifies
    1 + 5 = 6 rows per request.  The SM90 FP4 indexer consumes one row per
    query (the metadata builder is forced to flatten for it), and each row
    carries its own acceptance-dependent visible length.  A row's finite
    columns must depend only on that row's own context, so a shorter accepted
    prefix in one row must not leak into another, and an all-padding row must
    come back entirely ``-inf``.
    """
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(11)
    device = "cuda"
    page_size = PAGE_SIZE
    heads = 32
    num_blocks, cbs = 16, 8
    # One request's 6 verification rows: 5 draft positions + the bonus token,
    # with acceptance shrinking the visible length down the group.  The last
    # row is a pure padding row (idle rank / unused slot).
    n_vis_list = [100, 92, 77, 60, 33, 0]
    rows = len(n_vis_list)
    candidates = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1]
    width = len(candidates) * cbs

    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    context_lens = torch.tensor(n_vis_list, device=device, dtype=torch.int32)
    candidate_blocks = torch.tensor(candidates, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)

    compact = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    dense = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=num_blocks * page_size,
    )

    cols = torch.arange(width, device=device)
    logic = candidate_blocks[:, cols // cbs].to(torch.int64) * cbs + (cols % cbs).to(
        torch.int64
    )
    for r, n_vis in enumerate(n_vis_list):
        finite = compact[r] != float("-inf")
        mapped = logic[r][finite]
        assert mapped.numel() == n_vis, (
            f"row {r} (n_vis={n_vis}) exposes {mapped.numel()} finite columns"
        )
        if n_vis == 0:
            assert not finite.any()
            continue
        assert torch.equal(
            torch.sort(mapped).values, torch.arange(n_vis, device=device)
        )
        torch.testing.assert_close(
            compact[r][finite], dense[r][logic[r][finite]], rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# Group-6 K reuse (DSpark static target-verify shape)
# ---------------------------------------------------------------------------
#
# Admission is host-side (``query_group_size == 6`` + a full group).
# Request identity and, in compact mode, per-tile candidate-row
# equality are re-checked on device and fall back to per-row K reloads.  Every
# test below uses the *existing* per-row kernel as the oracle and demands
# ``rtol=0, atol=0``: the grouped kernel keeps each query's own packed Q load
# and per-head MMA, so it must be bit-identical, not merely close.


def _group6_dense_case(rows: int, n_vis: list[int], device, seed: int = 21):
    """Shared dense setup: one request's ``rows`` verify queries, one page table."""
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(seed)
    heads = 32
    page_size = PAGE_SIZE
    num_blocks = 16
    width = num_blocks * page_size
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    context_lens = torch.tensor(n_vis, device=device, dtype=torch.int32)
    return (
        mxfp4_paged_index_logits,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    )


@requires_sm90
def test_group6_matches_per_row_dense(monkeypatch):
    """A full group of six dense rows must equal six per-row calls bit-exactly."""
    device = "cuda"
    rows = 12  # two full groups
    n_vis = [300, 260, 220, 180, 120, 0, 305, 250, 205, 150, 90, 1]
    (
        fn,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    ) = _group6_dense_case(rows, n_vis, device)
    row_indices = torch.zeros(rows, device=device, dtype=torch.int32)

    grouped = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    # The all-invisible row (n_vis == 0) is entirely -inf in both.
    assert (grouped[5] == float("-inf")).all()
    assert (grouped[11, 1:] == float("-inf")).all()


@requires_sm90
def test_group6_dense_spanning_two_requests_falls_back(monkeypatch):
    """A group straddling two requests must fall back per row, never share K.

    This is the sharpest silent-regression detector for the device-side request
    identity guard: the row order is deliberately interleaved so every group
    mixes requests.  If the guard were dropped, each non-leader row would score
    against the leader's page table and differ from the per-row result.
    """
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(31)
    device = "cuda"
    rows, heads, page_size = 12, 32, PAGE_SIZE
    num_blocks = 32
    width = 16 * page_size
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    # Request 0 owns blocks 0..15, request 1 owns 16..31.
    block_table = torch.zeros((rows, 16), device=device, dtype=torch.int32)
    block_table[:6, :16] = torch.arange(16, device=device, dtype=torch.int32)
    block_table[6:, :16] = torch.arange(16, 32, device=device, dtype=torch.int32)
    context_lens = torch.full((rows,), 8 * page_size, device=device, dtype=torch.int32)
    # Interleaved request ids: every group mixes 0 and 1.
    row_indices = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.int32
    )

    grouped = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    # Sanity: the two requests really do read different K.
    assert not torch.equal(per_row[0], per_row[1])


@requires_sm90
def test_group6_partial_group_and_padding(monkeypatch):
    """Non-multiple-of-6 rows and padded/zero-visible rows stay exact.

    ``rows == 4`` cannot form a full group, so admission fails and the existing
    per-row grid runs; a six-row group whose last leader id is a padding
    sentinel falls back in-kernel.  Both must be byte-identical to per-row.
    """
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    device = "cuda"

    # (a) rows not a multiple of 6 -> host fallback.
    rows = 4
    (
        _,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    ) = _group6_dense_case(rows, [120, 90, 60, 0], device, seed=41)
    row_indices = torch.zeros(rows, device=device, dtype=torch.int32)
    grouped = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert torch.equal(grouped, per_row)

    # (b) full group, last row a padding sentinel -> in-kernel per-row fallback.
    rows = 6
    (
        _,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    ) = _group6_dense_case(rows, [200, 150, 120, 90, 60, 0], device, seed=42)
    row_indices = torch.tensor([0, 0, 0, 0, 0, 7], device=device, dtype=torch.int32)
    grouped = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    assert (grouped[5] == float("-inf")).all()


@requires_sm90
def test_group6_all_padding_zero_visible(monkeypatch):
    """The all-invisible group takes the dense early exit and writes all -inf."""
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    device = "cuda"
    rows = 6
    (
        _,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    ) = _group6_dense_case(rows, [0] * rows, device, seed=43)
    row_indices = torch.zeros(rows, device=device, dtype=torch.int32)
    grouped = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    assert (grouped == float("-inf")).all()


def _group6_compact_case(rows: int, n_vis: list[int], device, seed: int = 51):
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    torch.manual_seed(seed)
    heads = 32
    page_size = PAGE_SIZE
    num_blocks, cbs = 16, 8
    candidates = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1]
    width = len(candidates) * cbs
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    context_lens = torch.tensor(n_vis, device=device, dtype=torch.int32)
    candidate_blocks = torch.tensor(candidates, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)
    return (
        mxfp4_paged_index_logits,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        width,
    )


@requires_sm90
def test_group6_matches_per_row_compact(monkeypatch):
    """The compact extension must be exact when candidate rows agree per tile."""
    device = "cuda"
    rows = 6
    (
        fn,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        width,
    ) = _group6_compact_case(rows, [100, 92, 77, 60, 33, 0], device)
    row_indices = torch.zeros(rows, device=device, dtype=torch.int32)

    grouped = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
    )
    assert grouped.shape == (rows, width)
    assert torch.equal(grouped, per_row)


@requires_sm90
def test_group6_compact_mixed_candidates_falls_back(monkeypatch):
    """A compact tile whose candidate rows disagree must fall back per row."""
    device = "cuda"
    rows = 6
    (
        fn,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        _width,
    ) = _group6_compact_case(rows, [100, 92, 77, 60, 33, 0], device)
    # Rows 0..2 keep the shared candidates; rows 3..5 differ in the *second*
    # block tile (block columns 8..15), so only that tile must fall back while
    # the first tile can still be shared.
    candidate_blocks[3:, 8:] = torch.tensor(
        [11, 10, 9, 8, 7, 6, -1, -1], device=device, dtype=torch.int32
    )
    row_indices = torch.zeros(rows, device=device, dtype=torch.int32)

    grouped = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
    )
    assert torch.equal(grouped, per_row)


def _group6_share_predicate(row_ids, first: int = 0) -> bool:
    """Python replica of the kernel's on-device request-identity reduction.

    ``_mxfp4_grouped_paged_index_logits_kernel`` reduces over ``PGROUP`` (8)
    lanes while only ``GROUP`` (6) are real rows::

        gmask = (offs < GROUP) & (rows < n_rows)
        reqs = tl.load(row_indices + rows, mask=gmask, other=0)
        share = tl.sum(((reqs != req0) & gmask).to(tl.int32), 0) == 0

    Keeping the predicate in Python is deliberate: the bug it pins (comparing
    the two padding lanes, whose ``other`` is 0) is invisible in the kernel's
    *output* -- a fallback group is numerically identical to a shared one, just
    slower -- and an in-kernel counter perturbs unrelated kernel tests, so the
    regression is pinned here, next to the GPU tests that pin the outputs.
    """
    group, pgroup = 6, 8
    n_rows = len(row_ids)
    req0 = row_ids[first]
    for off in range(pgroup):
        row = first + off
        if off >= group or row >= n_rows:
            continue  # the padding lane the kernel masks out
        if row_ids[row] != req0:
            return False
    return True


def test_group6_identity_predicate_ignores_reduction_padding_cpu():
    """Non-zero request ids must share; the padded lanes must not veto."""
    # A full group of one request shares whatever that request id is.
    for req_id in (0, 1, 7, 31):
        assert _group6_share_predicate([req_id] * 6), req_id
    # The old predicate (no ``& gmask``) is False for every non-zero id: that
    # is the regression, and it silently disabled K reuse batch-wide.
    for req_id in (1, 7, 31):
        old = sum(1 for o in range(8) if ([req_id] * 6 + [0, 0])[o] != req_id) == 0
        assert old is False, req_id
    # A group straddling two requests must fall back.
    assert not _group6_share_predicate([0, 0, 0, 7, 7, 7])
    assert not _group6_share_predicate([7, 7, 7, 0, 0, 0])
    # A partial last group still shares when every *real* row agrees: lanes
    # past ``n_rows`` are masked, exactly like lanes past ``GROUP``.
    assert _group6_share_predicate([5, 5, 5, 5])
    assert _group6_share_predicate([5, 5, 5, 3]) is False
    # The partial group is the *last* group of a longer batch.
    assert _group6_share_predicate([0, 0, 0, 0, 0, 0, 5, 5, 5, 5], first=6)
    assert _group6_share_predicate([0, 0, 0, 0, 0, 0, 5, 5, 5, 3], first=6) is False


@requires_sm90
@pytest.mark.parametrize("req_id", [0, 1, 7, 31])
def test_group6_shares_full_groups_with_any_request_id(monkeypatch, req_id):
    """A full group of one *non-zero* request id must still be bit-exact.

    The reduction-padding bug made ``share`` False for every non-zero request
    id. Check the request-identity predicate and compare logits with the
    per-row kernel for the same inputs.
    """
    device = "cuda"
    rows = 12  # two full groups
    n_vis = [300, 260, 220, 180, 120, 0, 305, 250, 205, 150, 90, 1]
    (
        fn,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    ) = _group6_dense_case(rows, n_vis, device)
    row_indices = torch.full((rows,), req_id, device=device, dtype=torch.int32)

    grouped = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    # The predicate that decides the branch, evaluated on the same inputs.
    assert _group6_share_predicate([req_id] * 6)


@requires_sm90
def test_group6_multi_request_batch_shares_every_full_group(monkeypatch):
    """Three single-request groups (ids 0/7/31) match the per-row kernel."""
    device = "cuda"
    rows = 18  # three full groups
    (
        fn,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        width,
    ) = _group6_dense_case(
        rows,
        [
            300,
            260,
            220,
            180,
            120,
            0,
            305,
            250,
            205,
            150,
            90,
            1,
            240,
            200,
            160,
            120,
            80,
            40,
        ],
        device,
    )
    row_indices = torch.tensor(
        [0] * 6 + [7] * 6 + [31] * 6, device=device, dtype=torch.int32
    )

    grouped = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    ids = row_indices.tolist()
    for start in (0, 6, 12):
        assert _group6_share_predicate(ids, first=start)


@requires_sm90
def test_group6_mixed_request_group_falls_back_bit_exactly(monkeypatch):
    """Interleaved request ids must keep the per-row fallback bit-exact."""
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    device = "cuda"
    rows, heads, page_size = 12, 32, PAGE_SIZE
    num_blocks = 32
    width = 16 * page_size
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.zeros((rows, 16), device=device, dtype=torch.int32)
    block_table[:6, :16] = torch.arange(16, device=device, dtype=torch.int32)
    block_table[6:, :16] = torch.arange(16, 32, device=device, dtype=torch.int32)
    context_lens = torch.full((rows,), 8 * page_size, device=device, dtype=torch.int32)
    # Every group mixes request 0 and request 7.
    row_indices = torch.tensor(
        [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7], device=device, dtype=torch.int32
    )

    grouped = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=width,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=width,
    )
    assert torch.equal(grouped, per_row)
    # Each group straddles requests 0 and 7 -> the fallback is required.
    ids = row_indices.tolist()
    for start in (0, 6):
        assert not _group6_share_predicate(ids, first=start)


@requires_sm90
def test_group6_compact_invisible_tiles_match_per_row(monkeypatch):
    """Invisible compact tiles retain the per-row kernel's -inf mask."""
    device = "cuda"
    rows = 6
    (
        fn,
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        width,
    ) = _group6_compact_case(rows, [100, 92, 77, 60, 33, 0], device)
    # Second tile (block columns 8..15) is invisible for *every* row: either an
    # invalid id or a logical position past the row's context length.
    candidate_blocks[:, 8:] = torch.tensor(
        [-1, 20, -1, 20, -1, 20, -1, 20], device=device, dtype=torch.int32
    )
    row_indices = torch.zeros(rows, device=device, dtype=torch.int32)

    grouped = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
        row_indices=row_indices,
        query_group_size=6,
    )
    per_row = fn(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
        row_indices=row_indices,
        query_group_size=1,
    )
    assert torch.equal(grouped, per_row)
    assert torch.isneginf(grouped[:, width // 2 :]).all()


def test_group6_candidate_tile_guard_is_necessary():
    """CPU form of the compact group guard (the kernel needs a GPU).

    Two rows may share a decoded K tile only if their candidate ids agree
    *tile-locally* (tile = ``BLOCK_L / cbs`` candidate blocks); otherwise the
    same compact column maps to different logical positions and the shared K is
    wrong.  Pins the guard's exact scope so it cannot be loosened to a
    whole-row comparison.
    """
    cbs, block_l = 8, 64
    blocks_per_tile = block_l // cbs
    cols = torch.arange(2 * block_l)
    base = torch.tensor([12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1])
    other = base.clone()
    # Differ only in the second tile's block columns.
    other[blocks_per_tile : 2 * blocks_per_tile] = torch.tensor(
        [11, 10, 9, 8, 7, 6, -1, -1]
    )

    logical_base = base[cols // cbs].to(torch.int64) * cbs + (cols % cbs)
    logical_other = other[cols // cbs].to(torch.int64) * cbs + (cols % cbs)

    tile0 = slice(0, block_l)
    tile1 = slice(block_l, 2 * block_l)
    # Tile 0 agrees: the shared tile is exact there.
    assert torch.equal(logical_base[tile0], logical_other[tile0])
    # Tile 1 disagrees: sharing it would score different logical positions.
    assert not torch.equal(logical_base[tile1], logical_other[tile1])


@requires_sm90
def test_compact_decode_empty_and_single_row_shapes():
    """Padding-only batches must not fault or read out of bounds.

    ``rows == 0`` is the idle-rank / no-decode-row case and ``topk_tokens``
    may exceed the compact width on a first step whose candidates are all
    padding.
    """
    from flag_gems.fused.DSA.mxfp4_mqa_logits import (
        mxfp4_paged_index_logits,
    )

    device = "cuda"
    page_size, heads, num_blocks, cbs = PAGE_SIZE, 32, 16, 8
    cache = _packed_cache(num_blocks, page_size, device)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )
    q_values = _rand_q(1, heads, device)
    q_scale = _valid_q_scale(1, heads, device)
    weights = torch.randn(1, heads, device=device, dtype=torch.bfloat16)
    context_lens = torch.zeros(1, device=device, dtype=torch.int32)
    # All-padding candidates, with no visible token at all.
    candidate_blocks = torch.full((1, 16), -1, device=device, dtype=torch.int32)

    logits = mxfp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    assert logits.shape == (1, 16 * cbs)
    assert (logits == float("-inf")).all()

    empty = mxfp4_paged_index_logits(
        q_values[:0],
        q_scale[:0],
        cache,
        weights[:0],
        context_lens[:0],
        block_table[:0],
        candidate_blocks[:0],
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    assert empty.shape == (0, 16 * cbs)
