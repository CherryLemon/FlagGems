# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flag_gems.fused.dsv41_reference_ops import (
    fp4_quantize_reference,
    hc_split_sinkhorn_reference,
    sparse_attention_with_sink,
)


@pytest.mark.parametrize("rows", [0, 1, 7, 33])
@pytest.mark.parametrize("iterations,eps", [(1, 1e-6), (20, 1e-6), (20, 1e-4)])
def test_hc_precise_reference_and_repeat(rows, iterations, eps):
    from flag_gems.fused.mhc.hc_split_sinkhorn import mhc_split_sinkhorn_torch_ref

    torch.manual_seed(41)
    mixes = torch.randn(1, rows, 24, device="cuda") * 4
    scale = torch.tensor([0.7, 1.3, 2.1], device="cuda")
    base = torch.randn(24, device="cuda")
    expected = mhc_split_sinkhorn_torch_ref(mixes, scale, base, 4, iterations, eps)
    actual = hc_split_sinkhorn_reference(mixes, scale, base, 4, iterations, eps)
    repeated = hc_split_sinkhorn_reference(mixes, scale, base, 4, iterations, eps)
    for a, e, r in zip(actual, expected, repeated):
        torch.testing.assert_close(a, e, atol=1e-6, rtol=2e-6)
        torch.testing.assert_close(a, r, atol=0, rtol=0)


@pytest.mark.parametrize("scale_format,group", [("e8m0", 32), ("e4m3", 16)])
@pytest.mark.parametrize("rows", [0, 1, 7])
def test_fp4_rounding_and_packing(scale_format, group, rows):
    torch.manual_seed(41)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    if rows:
        # Include every tie, its sign, zeros, and a scale-setting maximum.
        x[0, :16] = torch.tensor(
            [
                0,
                -0.0,
                0.25,
                -0.25,
                0.75,
                -0.75,
                1.25,
                -1.25,
                1.75,
                -1.75,
                2.5,
                -2.5,
                3.5,
                -3.5,
                5,
                6,
            ],
            device="cuda",
        )
    data = x.float().reshape(rows, 128 // group, group)
    maximum = data.abs().amax(-1)
    if scale_format == "e8m0":
        scale = torch.exp2(torch.ceil(torch.log2(maximum.clamp_min(6 * 2.0**-126) / 6)))
    else:
        scale = (maximum.clamp_min(6 * 2.0**-9) / 6).to(torch.float8_e4m3fn).float()
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    v = data / scale[..., None]
    distance = (v.abs()[..., None] - table).abs()
    best = distance.amin(-1, keepdim=True)
    # Prefer even code at an exact halfway boundary, independently of kernel thresholds.
    priority = torch.tensor([0, 9, 2, 11, 4, 13, 6, 15], device="cuda")
    codes = torch.where(distance == best, priority, 100).argmin(-1)
    expected = (
        (table[codes] * torch.where(v.signbit(), -1, 1) * scale[..., None])
        .reshape(x.shape)
        .to(x.dtype)
    )
    packed, scales = fp4_quantize_reference(x, group, scale_format=scale_format)
    torch.testing.assert_close(scales.float(), scale, rtol=0, atol=0)
    all_codes = torch.stack((packed & 15, packed >> 4), -1).reshape(data.shape)
    wanted = codes | (v.signbit().int() * 8)
    torch.testing.assert_close(all_codes.long(), wanted.long(), rtol=0, atol=0)
    result = fp4_quantize_reference(
        x.clone(), group, scale_format=scale_format, inplace=True
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def attention_reference(q, kv, sink, indices, scale):
    b, s, _h, _d = q.shape
    result = torch.zeros_like(q)
    for bi in range(b):
        for si in range(s):
            valid = indices[bi, si]
            valid = valid[(valid >= 0) & (valid < kv.shape[1])].long()
            values = kv[bi, valid].float()
            scores = q[bi, si].float() @ values.T * scale
            if valid.numel() == 0:
                continue
            maxima = scores.amax(-1, keepdim=True)
            prob = (scores - maxima).exp()
            # Published TileLang PV rounds unnormalized probabilities to BF16.
            numerator = prob.bfloat16().float() @ values
            denominator = prob.sum(-1, keepdim=True) + (sink[:, None] - maxima).exp()
            result[bi, si] = (numerator / denominator).to(q.dtype)
    return result


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 512, 131, 33), (2, 7, 16, 128, 17, 17), (1, 3, 32, 64, 1, 0)]
)
def test_sparse_sink_attention(shape):
    b, s, h, d, n, k = shape
    torch.manual_seed(1041)
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(h, device="cuda") * 3
    idx = torch.randint(-1, n, (b, s, k), device="cuda", dtype=torch.int32)
    if s > 1:
        idx[:, 0] = -1
    actual = sparse_attention_with_sink(q, kv, sink, idx, d**-0.5)
    expected = attention_reference(q, kv, sink, idx, d**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.016)


def test_graph_replay_updates_cache_and_q():
    q = torch.randn(1, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 16, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(8, device="cuda")
    idx = torch.arange(16, device="cuda", dtype=torch.int32).view(1, 1, 16)
    for _ in range(2):
        fp4_quantize_reference(kv, 16, scale_format="e4m3", inplace=True)
        sparse_attention_with_sink(q, kv, sink, idx, 512**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp4_quantize_reference(kv, 16, scale_format="e4m3", inplace=True)
        out = sparse_attention_with_sink(q, kv, sink, idx, 512**-0.5)
    kv.normal_()
    q.normal_()
    graph.replay()
    expected = attention_reference(q, kv, sink, idx, 512**-0.5)
    torch.testing.assert_close(out, expected, atol=0.004, rtol=0.016)
