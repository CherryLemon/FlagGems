# SPDX-License-Identifier: Apache-2.0
"""Source-branch kernel contracts: compact remap, software FP4 and native FP8."""

import pytest
import torch
import triton
import triton.language as tl

from flag_gems.fused.DSA.finalize_candidate_topk import finalize_candidate_topk
from flag_gems.fused.fused_indexer_q_rope_quant import _fp32x2_to_fp4x2

CUDA = torch.cuda.is_available() and torch.version.hip is None
pytestmark = pytest.mark.skipif(not CUDA, reason="this acceptance run requires NVIDIA")


@triton.jit
def pack_pairs(X, Y, O, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    x = tl.load(X + i, i < N, 0.0)
    y = tl.load(Y + i, i < N, 0.0)
    tl.store(O + i, _fp32x2_to_fp4x2(x, y, False), i < N)


def test_software_fp4_all_ties_neighbors_zero_and_saturation():
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    x = torch.cat(
        [
            torch.nextafter(mid, mid * 0),
            mid,
            torch.nextafter(mid, mid + 1),
            mid * 0,
            torch.tensor([0.0, -0.0, 6.0, 7.0], device="cuda"),
        ]
    )
    x = torch.cat([x, -x]).contiguous()
    y = x.flip(0).contiguous()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")

    def expected(t):
        d = (t.abs()[:, None] - grid).abs()
        near = d == d.amin(-1, keepdim=True)
        code = near.int().argmax(-1)
        # A halfway point selects the even code, independent of this Triton helper.
        code = torch.where((near.sum(-1) > 1) & (code % 2 == 1), code + 1, code)
        return (code | (((t < 0) & (code != 0)).long() << 3)).byte()

    out = torch.empty_like(x, dtype=torch.uint8)
    ref = expected(x) | (expected(y) << 4)
    for _ in range(10):
        pack_pairs[(1,)](x, y, out, x.numel(), triton.next_power_of_2(x.numel()))
        assert torch.equal(out, ref)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_finalizer_visibility_nan_inf_unsorted_candidates_and_strides(compact, raw):
    scores = torch.tensor(
        [
            [1.0, float("nan"), float("inf"), -float("inf"), 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        ],
        device="cuda",
    )
    selected = torch.tensor(
        [[0, 1, 2, 3, 4, 7, -1, 8], [7, 0, 6, 1, -1, 8, 2, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    candidates = (
        torch.tensor([[2, 0, -1, 1], [1, 2, 0, -1]], dtype=torch.int32, device="cuda")
        if compact
        else None
    )
    lens = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    table = torch.tensor([[7, 3, 9, 5], [2, 4, 8, 6]], dtype=torch.int32, device="cuda")
    output = torch.full((2, 16), 123, dtype=torch.int32, device="cuda")[:, :8]
    raw_output = torch.empty_like(output) if raw else None
    expected = torch.full_like(output, -1)
    expected_raw = torch.full_like(output, -1)
    # Literal scalar reference keeps the selected order and uses mapped logical
    # visibility, including +inf as a valid newest-block score.
    for r in range(2):
        for c in range(8):
            s = int(selected[r, c])
            if not 0 <= s < 8 or not bool(scores[r, s] > -torch.inf):
                continue
            logical = (
                s if candidates is None else int(candidates[r, s // 2]) * 2 + s % 2
            )
            if not 0 <= logical < int(lens[r]):
                continue
            expected_raw[r, c] = logical
            expected[r, c] = table[r, logical // 2] * 2 + logical % 2
    for _ in range(10):
        finalize_candidate_topk(
            selected,
            scores,
            lens,
            table,
            output,
            block_size=2,
            candidate_blocks=candidates,
            candidate_block_size=2,
            raw_indices=raw_output,
        )
        assert torch.equal(output, expected)
        if raw:
            assert torch.equal(raw_output, expected_raw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        finalize_candidate_topk(
            selected,
            scores,
            lens,
            table,
            output,
            block_size=2,
            candidate_blocks=candidates,
            candidate_block_size=2,
            raw_indices=raw_output,
        )
    graph.replay()
    assert torch.equal(output, expected)
    # Replay must read fresh visibility bounds, not capture the old lengths.
    lens.zero_()
    graph.replay()
    assert torch.all(output == -1)


def test_finalizer_empty_and_all_invalid():
    for rows, width in [(0, 0), (2, 0), (2, 8)]:
        selected = torch.full((rows, 8), -1, dtype=torch.int32, device="cuda")
        scores = torch.full((rows, width), -torch.inf, device="cuda")
        lens = torch.zeros(rows, dtype=torch.int32, device="cuda")
        table = torch.zeros(rows, 1, dtype=torch.int32, device="cuda")
        out = torch.empty_like(selected)
        finalize_candidate_topk(selected, scores, lens, table, out, block_size=2)
        assert torch.equal(out, selected)


@pytest.mark.parametrize(
    "m,n,k", [(1, 33, 32), (5, 65, 96), (17, 64, 128), (7, 1280, 5120)]
)
@pytest.mark.parametrize("split,swap", [(1, False), (2, False), (4, True)])
def test_static_fp8_block_scales_split_swap_and_repeated_output(m, n, k, split, swap):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native static GEMM is Hopper only")
    from flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul_static import (
        sm90_static_gemm,
    )

    torch.manual_seed(31)
    # Exact small integers isolate layout/scale/reduction errors from input QDQ.
    a = torch.randint(-4, 5, (m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (n, k), device="cuda").to(torch.float8_e4m3fn)
    a_s = torch.exp2(torch.randint(-2, 2, (m, k // 32), device="cuda").float())
    b_s = torch.randint(125, 129, (n, k // 32), dtype=torch.uint8, device="cuda")
    bs_float = torch.exp2(b_s.float() - 127)
    expected = torch.zeros(m, n, device="cuda")
    for j in range(k // 32):
        tile = (
            a[:, j * 32 : (j + 1) * 32].float() @ b[:, j * 32 : (j + 1) * 32].float().T
        )
        expected += tile * a_s[:, j, None] * bs_float[None, :, j]
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": split,
        "SWAP_AB": swap,
        "num_warps": 4,
        "num_stages": 2,
    }
    for _ in range(10):
        actual = sm90_static_gemm(a, b, a_s, b_s, config, out_dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sm90_static_gemm(a, b, a_s, b_s, config, out_dtype=torch.float32)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 6, 7, 32, 33])
def test_static_fp8_tuned_dispatch_and_bf16_output(m):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native static GEMM is Hopper only")
    from flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul_static import (
        select_sm90_static_config,
        sm90_static_gemm,
    )

    # Checkpoint wq_a shape and DSpark/small-prefill M switch boundaries.
    n, k = 1280, 5120
    torch.manual_seed(17)
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.float8_e4m3fn)
    a_s = torch.ones(m, k // 32, device="cuda")
    b_s = torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8)
    expected = (a.float() @ b.float().T).bfloat16()
    assert select_sm90_static_config(n, k, m)["BLOCK_SIZE_K"] == 32
    actual = sm90_static_gemm(a, b, a_s, b_s)
    assert torch.equal(actual, expected)
