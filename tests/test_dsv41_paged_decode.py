# SPDX-License-Identifier: Apache-2.0
"""Page-aware FlagGems kernels must survive metadata changes during replay."""

import pytest
import torch

from flag_gems.fused.dsv41_decode_state import paged_index_scores, write_request_rows
from flag_gems.fused.dsv41_reference_ops import (
    paged_sparse_attention_with_sink,
    sparse_attention_with_sink,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_masked_state_write_replays_to_different_pages_and_positions():
    storage = torch.zeros(5, 71, 64, device="cuda", dtype=torch.bfloat16)
    pool = storage[:, :65]
    values = torch.randn(3, 2, 64, device="cuda", dtype=pool.dtype)
    pages = torch.tensor([3, 1, 4], device="cuda")
    positions = torch.tensor([[0, 64], [31, 32], [4, 5]], device="cuda")
    active = torch.tensor([[True, False], [True, True], [False, False]], device="cuda")
    write_request_rows(pool, values, pages, positions, active)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_request_rows(pool, values, pages, positions, active)
    expected = torch.zeros_like(storage)
    for order, slots in (
        ([3, 1, 4], [[0, 64], [31, 32], [4, 5]]),
        ([2, 4, 1], [[64, 0], [63, 64], [1, 2]]),
    ):
        storage.zero_()
        expected.zero_()
        pages.copy_(torch.tensor(order, device="cuda"))
        positions.copy_(torch.tensor(slots, device="cuda"))
        values.add_(1)
        graph.replay()
        for b, page in enumerate(order):
            for s, pos in enumerate(slots[b]):
                if active[b, s]:
                    expected[page, pos] = values[b, s]
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads,dim", [(4, 64), (16, 128), (32, 512)])
def test_paged_attention_matches_original_and_replays(heads, dim):
    torch.manual_seed(41)
    pages = torch.tensor([3, 1, 4], device="cuda")
    window = torch.randn(5, 16, dim, device="cuda", dtype=torch.bfloat16)[:, :8]
    compressed = torch.randn(5, 149, dim, device="cuda", dtype=window.dtype)[:, :137]
    query = torch.randn(3, 2, heads, dim, device="cuda", dtype=window.dtype)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    indices = torch.randint(-1, 145, (3, 2, 73), device="cuda", dtype=torch.int32)
    indices[1].fill_(-1)

    def forward():
        return paged_sparse_attention_with_sink(
            query, window, compressed, sink, indices, pages, dim**-0.5
        )

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for order in ([3, 1, 4], [2, 4, 0]):
        pages.copy_(torch.tensor(order, device="cuda"))
        query.mul_(0.9)
        graph.replay()
        reference = sparse_attention_with_sink(
            query,
            torch.cat([window[pages], compressed[pages]], 1),
            sink,
            indices,
            dim**-0.5,
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert actual[1].count_nonzero() == 0


@pytest.mark.parametrize("heads", [4, 16, 32])
def test_paged_index_scores_match_bf16_rounding(heads):
    torch.manual_seed(54)
    q = torch.randn(3, 2, heads, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(5, 151, 128, device="cuda", dtype=q.dtype)[:, :139]
    weights = torch.randn(3, 2, heads, device="cuda", dtype=q.dtype)
    pages = torch.tensor([4, 0, 2], device="cuda")
    lengths = torch.tensor([[1, 2], [127, 128], [138, 139]], device="cuda")
    paged_index_scores(q, keys, weights, pages, lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = paged_index_scores(q, keys, weights, pages, lengths)
    for order in ([4, 0, 2], [3, 1, 0]):
        pages.copy_(torch.tensor(order, device="cuda"))
        q.mul_(0.9)
        graph.replay()
        ref = (
            torch.einsum("bshd,bnd->bshn", q, keys[pages]).relu_()
            * weights.unsqueeze(-1)
        ).sum(2)
        ref.masked_fill_(
            torch.arange(139, device="cuda") >= lengths.unsqueeze(-1), -torch.inf
        )
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
