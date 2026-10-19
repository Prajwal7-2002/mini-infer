import pytest
import torch

from mini_infer.cache.block_pool import BlockPool, BlockTable


def test_block_pool_allocation():
    pool = BlockPool(
        num_blocks=10,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert pool.num_free == 10

    # Allocate 3 blocks
    b0 = pool.allocate_block()
    b1 = pool.allocate_block()
    b2 = pool.allocate_block()
    assert pool.num_free == 7
    assert len({b0, b1, b2}) == 3

    # Free one block
    pool.free_block(b1)
    assert pool.num_free == 8

    # Free remaining
    pool.free_blocks([b0, b2])
    assert pool.num_free == 10


def test_block_pool_exhaustion():
    pool = BlockPool(
        num_blocks=2,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    _ = pool.allocate_block()
    _ = pool.allocate_block()
    assert pool.num_free == 0

    with pytest.raises(MemoryError):
        pool.allocate_block()


def test_block_table_growth_and_free():
    pool = BlockPool(
        num_blocks=10,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    table = BlockTable(block_size=16)

    # First 16 tokens should fit into 1 block
    for i in range(16):
        block_id, offset = table.append_slot(pool)
        assert offset == i

    assert len(table.physical_block_ids) == 1
    assert pool.num_free == 9
    assert table.num_tokens == 16

    # 17th token triggers allocation of a 2nd physical block
    block_id_2, offset_2 = table.append_slot(pool)
    assert offset_2 == 0
    assert len(table.physical_block_ids) == 2
    assert pool.num_free == 8
    assert table.num_tokens == 17

    # Freeing table releases all blocks back to the pool
    table.free(pool)
    assert pool.num_free == 10
    assert len(table.physical_block_ids) == 0
    assert table.num_tokens == 0
