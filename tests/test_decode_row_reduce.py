# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flag_gems.fused.decode_row_reduce import decode_row_reduce


def reference(x, mean, keepdim=False):
    return torch.cat(
        [
            part.mean(-1, keepdim=keepdim) if mean else part.sum(-1, keepdim=keepdim)
            for part in x.split(1)
        ]
    )


@pytest.mark.parametrize("columns", [128, 132, 256, 512, 516, 1280, 5120, 20480, 32768])
@pytest.mark.parametrize("rows", [1, 3, 6, 16, 32])
@pytest.mark.parametrize("mean", [True, False])
def test_reference_association(columns, rows, mean):
    torch.manual_seed(927)
    x = torch.randn(7, rows, columns, device="cuda", dtype=torch.float32)
    for values in (x, x.to(torch.bfloat16).float().square()):
        torch.testing.assert_close(
            decode_row_reduce(values, reference_rows=rows, mean=mean, keepdim=True),
            reference(values, mean, keepdim=True),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("mean", [True, False])
def test_graph_replay_refreshes_rows_and_preserves_special_values(mean):
    x = torch.zeros(20, 6, 5120, device="cuda", dtype=torch.float32)
    decode_row_reduce(x, reference_rows=6, mean=mean)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = decode_row_reduce(x, reference_rows=6, mean=mean)
    for seed in (11, 23, 57):
        torch.manual_seed(seed)
        x.normal_()
        x[0, 0, :4] = torch.tensor([1e20, 1, -1e20, 0], device="cuda")
        x[1, 1, 1] = float("inf")
        x[2, 2, :2] = torch.tensor([float("inf"), -float("inf")], device="cuda")
        x[3, 3, 1] = float("nan")
        graph.replay()
        torch.testing.assert_close(
            out, reference(x, mean), rtol=0, atol=0, equal_nan=True
        )


def test_empty_and_unsupported_geometry():
    assert decode_row_reduce(torch.empty(0, 128, device="cuda")).shape == (0,)
    for x in (
        torch.empty(2, 127, device="cuda"),
        torch.empty(2, 512, device="cuda", dtype=torch.bfloat16),
        torch.empty(2, 1024, device="cuda")[:, ::2],
        torch.empty(2 * 512 + 1, device="cuda")[1:].view(2, 512),
    ):
        with pytest.raises(ValueError):
            decode_row_reduce(x)
