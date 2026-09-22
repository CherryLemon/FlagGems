# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 reference for the per-32 scaled reduction contract."""

import pytest
import torch

from flag_gems.fused.act_quant import act_quant_triton
from flag_gems.fused.block_scaled_lowp_linear import block_scaled_lowp_linear


def quantize_reference(x):
    grouped = x.float().reshape(-1, x.shape[-1] // 32, 32)
    raw = grouped.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    scale = torch.pow(2.0, torch.ceil(torch.log2(raw)))
    q = (grouped / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), scale


def reference(x, w, s, fmt):
    q, a_scale = quantize_reference(x)
    n, k = w.shape[0], x.shape[-1]
    if fmt == "mxfp4":
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=x.device,
        )
        codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k).long()
        decoded = table[codes]
        weight_scale = s.float()
    else:
        decoded = w.float()
        weight_scale = s.float().repeat_interleave(32, 0)[:n]
    result = torch.zeros(x.shape[0], n, dtype=torch.float32, device=x.device)
    for group in range(k // 32):
        part = (
            q[:, group * 32 : (group + 1) * 32].float()
            @ decoded[:, group * 32 : (group + 1) * 32].T
        )
        result += (
            part * a_scale[:, group : group + 1] * weight_scale[:, group].unsqueeze(0)
        )
    return result


@pytest.mark.parametrize("fmt", ["mxfp4", "fp8"])
@pytest.mark.parametrize(
    "shape", [(0, 31, 64), (1, 31, 96), (7, 65, 128), (33, 32, 160)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_scaled_products_match_reference(shape, fmt, dtype):
    torch.manual_seed(231)
    m, n, k = shape
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    if fmt == "mxfp4":
        # All 16 E2M1 bit patterns occur, including the two zero encodings.
        w = (
            torch.arange(n * k // 2, device="cuda")
            .remainder(256)
            .to(torch.uint8)
            .reshape(n, k // 2)
        )
        sr = n
    else:
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        sr = (n + 31) // 32
    s = torch.randint(122, 130, (sr, k // 32), dtype=torch.uint8, device="cuda").view(
        torch.float8_e8m0fnu
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = reference(x, w, s, fmt).to(dtype)
        actual = block_scaled_lowp_linear(
            x, w, s, weight_format=fmt, output_dtype=dtype
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-4)


def test_quantization_retains_original_scale_rounding_and_e4m3_codes():
    # Include zeros, clipping-scale transitions and values below the amax floor.
    x = (
        torch.tensor(
            [0, 1e-6, 1e-4, 1, -1, 448, 449, -449], device="cuda", dtype=torch.bfloat16
        )
        .repeat(32)
        .reshape(4, 64)
    )
    actual, actual_scale = act_quant_triton(x, block_size=32, scale_fmt="ue8m0")
    expected, expected_scale = quantize_reference(x)
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


def test_graph_replay_observes_new_input_without_host_sync():
    x = torch.ones(1, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.full((32, 64), 0x32, device="cuda", dtype=torch.uint8)
    s = torch.full((32, 4), 125, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    block_scaled_lowp_linear(x, w, s, weight_format="mxfp4")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = block_scaled_lowp_linear(x, w, s, weight_format="mxfp4")
    graph.replay()
    before = out.clone()
    x.mul_(2)
    graph.replay()
    torch.testing.assert_close(out, before * 2, rtol=0, atol=0)
