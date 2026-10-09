# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed two-round CUDA Graph protocol; record samples with --record json."""

import statistics

import pytest
import torch

import flag_gems

from .conftest import update_result

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _timing(fn):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _compare(operator, shape, baseline, candidate, reference):
    speedups = []
    for number in (1, 2):
        order = (
            [("baseline", baseline), ("candidate", candidate)]
            if number == 1
            else [("candidate", candidate), ("baseline", baseline)]
        )
        row = {
            "operator": operator,
            "shape": shape,
            "round": number,
            "reference": reference,
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "method": "25 warmup calls; capture 100 calls; 5 CUDA-event samples",
        }
        for name, fn in order:
            row[name] = _timing(fn)
        row["speedup"] = row["baseline"]["median_us"] / row["candidate"]["median_us"]
        speedups.append(round(row["speedup"], 3))
        update_result(operator, row)
    print(operator, shape, "speedup", speedups, flush=True)


@pytest.mark.swiglu_oai
def test_swiglu_oai_cuda_graph():
    torch.manual_seed(59)
    for m, i in ((1, 384), (64, 768), (4096, 384), (5089, 768), (8192, 1536)):
        x = torch.randn((m, 2 * i), device="cuda", dtype=torch.bfloat16)

        def baseline():
            gate, up = x.chunk(2, -1)
            gate = gate.clamp(max=7.0)
            up = up.clamp(-7.0, 7.0)
            return (gate * torch.sigmoid(1.702 * gate)) * (up + 1.0)

        torch.testing.assert_close(
            flag_gems.swiglu_oai(x), baseline(), atol=0.032, rtol=0.016
        )
        _compare(
            "swiglu_oai",
            [m, i],
            baseline,
            lambda: flag_gems.swiglu_oai(x),
            "complete staged Torch chain, BF16",
        )


@pytest.mark.scaled_mm
def test_int8_scaled_mm_cuda_graph(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper opt-in tiles")
    torch.manual_seed(59)
    for m, k, n in (
        (1, 6144, 1536),
        (64, 6144, 1536),
        (4096, 6144, 1536),
        (5089, 6144, 3072),
        (8192, 1024, 6144),
        (4096, 1536, 6144),
        (5089, 6144, 768),
    ):
        a = torch.randint(-32, 32, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-32, 32, (k, n), device="cuda", dtype=torch.int8)
        sa = torch.rand((m, 1), device="cuda") * 0.01
        sb = torch.rand((1, n), device="cuda") * 0.01

        def baseline():
            monkeypatch.setenv("FLAGGEMS_I8_SCALED_MM_SHAPE_TILES", "0")
            return flag_gems.scaled_mm_int8(a, b, sa, sb, out_dtype=torch.bfloat16)

        def candidate():
            monkeypatch.setenv("FLAGGEMS_I8_SCALED_MM_SHAPE_TILES", "1")
            return flag_gems.scaled_mm_int8(a, b, sa, sb, out_dtype=torch.bfloat16)

        torch.testing.assert_close(candidate(), baseline(), rtol=0, atol=0)
        _compare(
            "scaled_mm_hopper_row_major_fallback",
            [m, k, n],
            baseline,
            candidate,
            "same generic Triton fallback, row-major B, existing autotuner",
        )
