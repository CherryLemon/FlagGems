# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from .int8_indexer_reference import quantize_reference
from .int8_indexer_reference_checks import (  # noqa: F401
    test_fp32_logarithm_precision_boundary_is_discriminating,
    test_nearest_even_signed_range_and_zero_scale,
    test_quantization_empty_and_power_of_two_thresholds,
)

pytestmark = pytest.mark.int8_indexer_quantize


@pytest.mark.int8_indexer_quantize
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_quantize_nearest_even_zero_and_scale_boundaries(dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_quantize

    x = torch.zeros(4, 128, dtype=dtype)
    x[0, :8] = torch.tensor([127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=dtype)
    x[1, 0] = 128
    raw = torch.nextafter(torch.tensor(2.0**-20), torch.tensor(float("inf")))
    x[3, 0] = (raw * 127).to(dtype)
    expected_q, expected_scale = quantize_reference(x)
    actual_q, actual_scale = int8_indexer_quantize(x.cuda())
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(actual_q.cpu(), expected_q, atol=0, rtol=0)
