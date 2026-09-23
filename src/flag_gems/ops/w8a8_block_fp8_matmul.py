# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
import math
import os
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl
import yaml

import flag_gems

logger = logging.getLogger(__name__)


def _get_default_w8a8_block_fp8_config(block_n: int, block_k: int) -> Dict[str, Any]:
    if flag_gems.device != "cuda":
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": min(128, block_k),
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 3,
        }

    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


@triton.jit
def w8a8_block_fp8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

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


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[Dict[int, Any]]:
    if not torch.cuda.is_available():
        logger.debug(
            "CUDA is unavailable on this backend; using default W8A8 block FP8 config."
        )
        return None

    device_name = torch.cuda.get_device_name().replace(" ", "_")
    file_name = f"fp8_w8a8-{block_n}-{block_k}.yaml"

    config_dir = os.path.join(os.path.dirname(__file__), "..", "utils", "configs")
    cfg_file = os.path.join(config_dir, file_name)

    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            logger.info(
                "Using config from %s for W8A8 block FP8 kernel.",
                cfg_file,
            )
            dev_data = yaml.safe_load(f).get(device_name, {})
            NK_data = dev_data.get(f"{N},{K}", {})

            result = {}
            for k, p in NK_data.items():
                # unpack the list into dictionary
                result[int(k)] = {
                    "BLOCK_SIZE_M": p[0],
                    "BLOCK_SIZE_N": p[1],
                    "BLOCK_SIZE_K": p[2],
                    "GROUP_SIZE_M": p[3],
                    "num_warps": p[4],
                    "num_stages": p[5],
                }
            if not result:
                return None
            return result

    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        cfg_file,
    )
    return None


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    if block_k < 32 or block_k & (block_k - 1):
        raise ValueError("FP8 K groups must be powers of two >= 32")
    if block_n <= 0:
        raise ValueError("block_n must be positive")
    if not As.is_floating_point() or not Bs.is_floating_point():
        raise TypeError("matmul requires numeric scales; decode UE8M0 bytes first")

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = math.prod(A.shape[:-1])

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    if M == 0 or N == 0:
        return C
    if K == 0:
        return C.zero_()

    if (
        flag_gems.device == "cuda"
        and A.ndim == 2
        and (block_n, block_k) == (32, 32)
        and A.dtype == B.dtype == torch.float8_e4m3fn
        and output_dtype in (torch.bfloat16, torch.float32)
        and torch.cuda.get_device_capability(A.device) == (9, 0)
    ):
        from ._w8a8_block_fp8_hopper import hopper_block32_config, matmul_hopper

        hopper_config = hopper_block32_config(M, N, K)
        if hopper_config is not None:
            return matmul_hopper(A, B, As, Bs, C, hopper_config)

    configs = get_w8a8_block_fp8_configs(N, K, block_n, block_k)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = _get_default_w8a8_block_fp8_config(block_n, block_k)

    # One K tile loads exactly one scale per operand. Tuned configurations,
    # like defaults, must not cross a quantization-group boundary. Copy the
    # cached configuration so calls with another group size cannot mutate it.
    config = dict(config)
    tile_k = min(config["BLOCK_SIZE_K"], block_k)
    if tile_k < 32 or block_k % tile_k:
        raise ValueError("GEMM K tile must divide the quantization group")
    config["BLOCK_SIZE_K"] = tile_k

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    w8a8_block_fp8_matmul_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C
