# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from .int8_indexer_reference import cache_store_reference
from .int8_indexer_reference_checks import (  # noqa: F401
    test_cache_page_bytes_and_slot_mutation,
)

pytestmark = pytest.mark.int8_indexer_cache_store


@pytest.mark.int8_indexer_cache_store
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_int8_indexer_cache_state(page_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_store

    cpu_keys = (torch.arange(4 * 135).reshape(4, 135) % 127 - 63).to(dtype)[:, :128]
    cpu_cache = torch.full((3, page_size, 132), 29, dtype=torch.uint8)
    slots = torch.tensor(
        [page_size - 1, page_size, -1, 2 * page_size + 1], dtype=torch.int64
    )
    expected = cache_store_reference(cpu_keys, cpu_cache.clone(), slots, page_size)
    keys = torch.empty_strided(cpu_keys.shape, (135, 1), dtype=dtype, device="cuda")
    keys.copy_(cpu_keys)
    cache = cpu_cache.cuda()
    result = int8_indexer_cache_store(keys, cache, slots.cuda(), page_size)
    assert result is None
    assert torch.equal(cache.cpu(), expected)
