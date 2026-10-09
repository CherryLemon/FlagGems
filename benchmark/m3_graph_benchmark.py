# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic two-round CUDA Graph benchmark; invoke as a standalone script."""

import argparse
import json
import os
import pathlib
import statistics
import time

import torch

import flag_gems

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    type=pathlib.Path,
    default=pathlib.Path("m3-operator-graph-results.json"),
)
OUTPUT = parser.parse_args().output
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(59)
torch.backends.cuda.matmul.allow_tf32 = False


def timing(fn):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            fn()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def compare(op, shape, baseline, candidate, reference):
    rows = []
    for number in (1, 2):
        order = (
            [("baseline", baseline), ("candidate", candidate)]
            if number == 1
            else [("candidate", candidate), ("baseline", baseline)]
        )
        row = {"operator": op, "shape": shape, "round": number, "reference": reference}
        for name, fn in order:
            row[name] = timing(fn)
        row["speedup"] = row["baseline"]["median_us"] / row["candidate"]["median_us"]
        rows.append(row)
        results["measurements"].append(row)
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(op, shape, "speedup", [round(row["speedup"], 3) for row in rows], flush=True)


results = {
    "gpu": torch.cuda.get_device_name(),
    "torch_version": torch.__version__,
    "method": "CUDA Graph; 25 warmup calls; capture 100 calls; 5 event samples; two rounds with reversed order",
    "measurements": [],
}
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
    compare(
        "swiglu_oai",
        [m, i],
        baseline,
        lambda: flag_gems.swiglu_oai(x),
        "complete staged Torch chain, BF16",
    )
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
        os.environ["FLAGGEMS_I8_SCALED_MM_SHAPE_TILES"] = "0"
        return flag_gems.scaled_mm_int8(a, b, sa, sb, out_dtype=torch.bfloat16)

    def candidate():
        os.environ["FLAGGEMS_I8_SCALED_MM_SHAPE_TILES"] = "1"
        return flag_gems.scaled_mm_int8(a, b, sa, sb, out_dtype=torch.bfloat16)

    actual = candidate()
    expected = baseline()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    compare(
        "scaled_mm_hopper_row_major_fallback",
        [m, k, n],
        baseline,
        candidate,
        "same generic Triton fallback, row-major B, existing autotuner",
    )
    os.environ.pop("FLAGGEMS_I8_SCALED_MM_SHAPE_TILES", None)

results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
