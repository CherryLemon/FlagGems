# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems
from flag_gems.fused.bf16_small_m_linear import bf16_small_m_linear


def _require_sm90():
    if flag_gems.vendor_name != "nvidia" or not torch.cuda.is_available():
        pytest.skip("SM90 CUDA specialization")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA specialization")


@pytest.mark.bf16_small_m_linear
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
@torch.no_grad()
def test_bf16_small_m_linear_math_and_ordinary_graph(rows):
    _require_sm90()
    generator = torch.Generator().manual_seed(3407 + rows)
    cpu_a = (torch.randn((rows, 6144), generator=generator) * 0.125).bfloat16()
    cpu_b = (torch.randn((rows, 6144), generator=generator) * 0.125).bfloat16()
    cpu_w = (torch.randn((1024, 6144), generator=generator) * 0.125).bfloat16()
    x = cpu_a.to(flag_gems.device)
    weight = cpu_w.to(flag_gems.device)
    out = torch.empty((rows, 1024), dtype=x.dtype, device=x.device)
    for _ in range(2):
        bf16_small_m_linear(x, weight, device_sm=90, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = bf16_small_m_linear(x, weight, device_sm=90, out=out)
    for cpu in (cpu_a, cpu_b, cpu_a):
        x.copy_(cpu)
        eager = bf16_small_m_linear(x, weight, device_sm=90)
        expected = (cpu.double() @ cpu_w.double().T).float()
        graph.replay()
        torch.testing.assert_close(graph_out, eager, atol=0, rtol=0)
        torch.testing.assert_close(
            graph_out.cpu().float(), expected, atol=0.015625, rtol=0.015625
        )


@pytest.mark.bf16_small_m_linear
@torch.no_grad()
def test_bf16_small_m_linear_rejects_unsupported_rows():
    _require_sm90()
    x = torch.empty((17, 6144), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.empty((1024, 6144), dtype=x.dtype, device=x.device)
    with pytest.raises(ValueError, match="unsupported"):
        bf16_small_m_linear(x, weight, device_sm=90)
