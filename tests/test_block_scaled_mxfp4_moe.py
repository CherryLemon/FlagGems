# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from flag_gems.fused.block_scaled_lowp_linear import block_scaled_lowp_linear
from flag_gems.fused.block_scaled_mxfp4_moe import block_scaled_mxfp4_moe


def _reference(x, ids, weights, gate_up, gate_up_scale, down, down_scale, start):
    m, hidden = x.shape
    intermediate = down.shape[-1] * 2
    result = torch.zeros((m, hidden), dtype=torch.float32, device=x.device)
    for expert_id in range(start, start + gate_up.shape[0]):
        token, slot = torch.where(ids == expert_id)
        if token.numel() == 0:
            continue
        local = expert_id - start
        inputs = x[token].contiguous()
        gate = block_scaled_lowp_linear(
            inputs, gate_up[local, :intermediate],
            gate_up_scale[local, :intermediate], weight_format="mxfp4",
        ).float()
        up = block_scaled_lowp_linear(
            inputs, gate_up[local, intermediate:],
            gate_up_scale[local, intermediate:], weight_format="mxfp4",
        ).float()
        gate = gate.clamp(max=10.0)
        up = up.clamp(min=-10.0, max=10.0)
        activated = (F.silu(gate) * up * weights[token, slot, None]).to(x.dtype)
        output = block_scaled_lowp_linear(
            activated.contiguous(), down[local], down_scale[local],
            weight_format="mxfp4",
        )
        result[token] += output.float()
    return result


@pytest.mark.parametrize("m,top_k", [(1, 3), (3, 3)])
def test_routed_mxfp4_matches_reference_and_graph_replay(m, top_k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(19)
    device = "cuda"
    hidden, intermediate, local_count, start = 128, 64, 3, 4
    gate_up = torch.randint(
        0, 256, (local_count, 2 * intermediate, hidden // 2),
        device=device, dtype=torch.uint8,
    )
    down = torch.randint(
        0, 256, (local_count, hidden, intermediate // 2),
        device=device, dtype=torch.uint8,
    )
    gate_scale = torch.full(
        (local_count, 2 * intermediate, hidden // 32), 127,
        device=device, dtype=torch.uint8,
    )
    down_scale = torch.full(
        (local_count, hidden, intermediate // 32), 127,
        device=device, dtype=torch.uint8,
    )
    x = torch.randn((m, hidden), device=device, dtype=torch.bfloat16)
    ids = torch.tensor(
        [[6, 1, 4], [5, 8, 1], [1, 2, 3]][:m],
        device=device, dtype=torch.int64,
    )
    weights = torch.rand((m, top_k), device=device, dtype=torch.float32)

    def run():
        return block_scaled_mxfp4_moe(
            x, ids, weights, gate_up, gate_scale, down, down_scale,
            local_start=start, swiglu_limit=10.0,
        )

    actual = run()
    expected = _reference(x, ids, weights, gate_up, gate_scale, down, down_scale, start)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        captured = run()
    for replay in range(2):
        x.copy_(torch.randn_like(x))
        ids.copy_(torch.tensor(
            [[4, 5, 0], [2, 6, 1], [5, 1, 4]][:m], device=device,
        ))
        weights.copy_(torch.rand_like(weights))
        graph.replay()
        expected = _reference(x, ids, weights, gate_up, gate_scale, down, down_scale, start)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
