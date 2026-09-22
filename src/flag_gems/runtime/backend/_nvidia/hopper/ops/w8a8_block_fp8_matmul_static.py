# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Hopper block-32 FP8 GEMM with deterministic split-K reduction.

Ported from CherryLemon/vllm a9e3d217cce075c77a8041d14dd822307953735e,
model_executor/kernels/linear/mxfp8/sm90_static.py (originally SGLang).
The branch's shape table, per-K32 FP8 dot, SWAP_AB, separate split-K
partials and reduction order are preserved. No vLLM runtime is imported.

A is E4M3 [M,K], B is E4M3 [N,K], As is FP32 [M,K/32], and Bs is
raw E8M0 bytes [N,K/32], with checkpoint N/32 scales replicated per row.
This optional API requires native FP8 on SM90. The portable compatibility
profile continues to use block_scaled_lowp_linear; this is not its fallback.
"""

import torch
import triton
import triton.language as tl

MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8

_MXFP8_BLOCK = 32

# ---------------------------------------------------------------------------
# Tuned (N, K) -> {M: config} table.
#
# Ported from the eleven H100 JSONs shipped by SGLang:
#   configs/N=<N>,K=<K>,device_name=NVIDIA_H100_80GB_HBM3,
#           dtype=fp8_w8a8,block_shape=[32, 32].json
# The M grid is irregular and every kernel-switch risk boundary has paired
# ``x``/``x + 1`` keys so that a nearest-M lookup cannot snap past a switch.
# DeepSeek-V4.1-Flash (hidden 5120) actually hits:
#   (1280, 5120)  wq_a
#   (512,  5120)  wkv
#   (4096, 1280)  wq_b @ TP8 (32768/8) and indexer.wq_b
#   (25600, 6144) engram.wkv
#   (5120, 15360) MTP main_proj
# The remaining table entries ((1536/1792/576, 5120), (16384, 1280),
# (5120, 288), (5120, 4096)) cover the same family at other TP/sizes.
# ---------------------------------------------------------------------------
# fmt: off
_SM90_STATIC_CONFIGS: dict[tuple[int, int], dict[int, dict]] = {
    (1280, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (1536, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (16384, 1280): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        44: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        45: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        64: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        96: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        384: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        385: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        768: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        769: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (1792, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        64: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        80: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        81: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        96: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        384: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        385: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        768: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        769: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (25600, 6144): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        64: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        96: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (4096, 1280): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        32: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        33: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (512, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (5120, 15360): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (5120, 288): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (5120, 4096): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        44: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        45: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        64: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        320: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        321: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        384: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        385: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        768: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        769: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (576, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        64: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        128: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        192: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        256: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        384: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        385: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        1536: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        1537: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        3072: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        3073: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
}
# fmt: on

# For (N, K) without a tuned entry. SWAP_AB/SplitK are off: correctness first,
# and any N is handled by the store mask.
_SM90_GENERIC_CONFIG: dict = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
    "SWAP_AB": False,
    "SPLIT_K": 1,
}


def select_sm90_static_config(N: int, K: int, M: int) -> dict:
    """Pick the tuned tile config for (N, K, M), else the generic fallback.

    ``M`` only selects *which* verified tile config runs; it is deliberately
    not a kernel specialization key (see ``sm90_static_gemm``), so an unseen
    prefill tail length or mixed-batch size reuses an existing binary instead
    of triggering a fresh JIT compile.  Mirrors SGLang's
    ``configs[min(configs, key=|M - key|)]`` lookup for the tile choice.
    """
    table = _SM90_STATIC_CONFIGS.get((N, K))
    if table is None:
        return dict(_SM90_GENERIC_CONFIG)
    chosen = table[min(table.keys(), key=lambda key: abs(key - M))]
    config = dict(chosen)
    config.setdefault("SWAP_AB", False)
    config.setdefault("SPLIT_K", 1)
    return config


# ---------------------------------------------------------------------------
# Verbatim port of SGLang fp8_hopper_static.py, with the uint8 ue8m0 decode of
# ``Bs`` added in-register (see module docstring).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["M"])
def _w8a8_block_fp8_matmul_hopper_static(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    split = tl.program_id(axis=1)
    tiles_per_split = tl.cdiv(tl.cdiv(K, BLOCK_SIZE_K), SPLIT_K)
    first_tile = split * tiles_per_split
    C += split * M * N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    a_ptrs += first_tile * BLOCK_SIZE_K * stride_ak
    b_ptrs += first_tile * BLOCK_SIZE_K * stride_bk
    As_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_As_k
    Bs_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k

    # Small-M Hopper configs transpose the MMA so the weight tile occupies M.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(
        first_tile,
        tl.minimum(first_tile + tiles_per_split, tl.cdiv(K, BLOCK_SIZE_K)),
        loop_unroll_factor=1,
    ):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        # vLLM keeps the weight scale as uint8 ue8m0 bytes; decode to the exact
        # fp32 value of 2**(exp-127) with the same bit trick as
        # fp8_utils._upcast_e8m0_to_fp32.
        b_s_raw = tl.load(Bs_ptrs)
        b_s = (b_s_raw.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if SWAP_AB:
            accumulator += (
                tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]
            )
        else:
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if SWAP_AB:
        accumulator = tl.trans(accumulator)

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(do_not_specialize=["ELEMENTS"])
def _reduce_block_fp8_split_k(
    Parts, Out, ELEMENTS, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    """Sum the ``SPLITS`` fp32 partials of the SplitK GEMM into ``Out``.

    ``ELEMENTS == M * N`` is a *runtime* argument: leaving it as a constexpr
    made every new M compile another reduction binary, defeating the
    ``do_not_specialize`` on the main GEMM for any config with ``SPLIT_K > 1``
    (which includes several tuned ``(N, K)`` entries).  Only the split count
    and the block size stay compile-time here.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        Parts + splits[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS,
        0.0,
    )
    tl.store(Out + offsets, tl.sum(values, axis=0), offsets < ELEMENTS)


def _contiguous_2d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def sm90_static_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    config: dict | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Launch the static block-32 GEMM with an explicit config.

    A:  [M, K] E4M3, contiguous, ``K % 32 == 0``.
    B:  [N, K] E4M3 (the vLLM weight layout; ``B.T`` is the [K, N] operand).
    As: [M, K // 32] fp32 per-row activation scale.
    Bs: [N, K // 32] uint8 ue8m0 per-row weight scale.

    ``M`` is passed as a *runtime* argument (``do_not_specialize``): the
    kernel's ``N``/``K``/tile parameters are ``tl.constexpr``, but M is not a
    valid specialization key here.  vLLM feeds this GEMM an open-ended set of
    token counts (every eager prefill tail, every mixed-batch size), so a
    constexpr M would compile a new binary per shape and make the first
    request of each new bucket pay a JIT spike that CUDA-graph capture cannot
    absorb.  Bounding compilation to the finite (N, K) config table keeps the
    verified tile tuning while removing the shape explosion.
    """
    if (
        A.device.type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(A.device)[0] != 9
    ):
        raise ValueError("sm90_static_gemm requires NVIDIA Hopper native FP8")
    assert A.ndim == B.ndim == As.ndim == Bs.ndim == 2
    assert A.dtype == MXFP8_VALUE_DTYPE and B.dtype == MXFP8_VALUE_DTYPE
    assert A.stride(-1) == 1, "A groups must be contiguous"
    assert As.dtype == torch.float32
    assert Bs.dtype == MXFP8_SCALE_DTYPE, (
        f"SM90 static kernel expects {MXFP8_SCALE_DTYPE} weight_scale, got {Bs.dtype}"
    )
    M, K = A.shape
    N = B.shape[0]
    assert K > 0 and K % 32 == 0 and B.shape[1] == K
    assert As.shape == (M, K // 32) and Bs.shape == (N, K // 32)
    assert all(t.device == A.device for t in (B, As, Bs))
    assert out_dtype in (torch.bfloat16, torch.float16, torch.float32)
    config = select_sm90_static_config(N, K, M) if config is None else dict(config)
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    split_k = int(config.get("SPLIT_K", 1))
    swap_ab = bool(config.get("SWAP_AB", False))
    assert split_k > 0 and split_k & (split_k - 1) == 0, (
        "SPLIT_K must be a power of two"
    )
    assert block_k == 32, "scale groups must be applied at every K32 dot"

    out = torch.empty((M, N), device=A.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return out
    partials = (
        torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
        if split_k > 1
        else out
    )
    needs_masking = bool(K % block_k != 0)
    grid = (
        triton.cdiv(M, block_m) * triton.cdiv(N, block_n),
        split_k,
    )
    _w8a8_block_fp8_matmul_hopper_static[grid](
        A,
        B,
        partials,
        As,
        Bs,
        M,
        N,
        K,
        1,  # group_n: vLLM scale is per-output-row (see module docstring)
        _MXFP8_BLOCK,  # group_k
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        partials.stride(-2),
        partials.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        needs_masking=needs_masking,
        SWAP_AB=swap_ab,
        SPLIT_K=split_k,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    if split_k > 1:
        _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](
            partials, out, M * N, split_k, 256
        )
    return out
