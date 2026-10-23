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


def test_block_pool_paged_read_write():
    pool = BlockPool(
        num_blocks=4,
        block_size=16,
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    b0 = pool.allocate_block()
    b1 = pool.allocate_block()

    # Write single slot
    k_vec = torch.randn(2, 8)
    v_vec = torch.randn(2, 8)
    pool.write_slot(layer_idx=0, block_id=b0, slot_offset=0, k_slot=k_vec, v_slot=v_vec)

    # Read back single token
    k_read, v_read = pool.read_sequence_kv(layer_idx=0, physical_block_ids=[b0], num_tokens=1)
    assert k_read.shape == (1, 2, 1, 8)
    assert torch.allclose(k_read[0, :, 0, :], k_vec)
    assert torch.allclose(v_read[0, :, 0, :], v_vec)

    # Test gather_batched_paged_kv for 2 sequences of different lengths
    k_vec2 = torch.randn(2, 8)
    v_vec2 = torch.randn(2, 8)
    pool.write_slot(layer_idx=0, block_id=b1, slot_offset=0, k_slot=k_vec2, v_slot=v_vec2)
    pool.write_slot(layer_idx=0, block_id=b1, slot_offset=1, k_slot=k_vec2, v_slot=v_vec2)

    block_tables = [[b0], [b1]]
    context_lens = [1, 2]
    k_batch, v_batch, mask = pool.gather_batched_paged_kv(
        layer_idx=0, block_tables=block_tables, context_lens=context_lens
    )

    assert k_batch.shape == (2, 2, 2, 8)
    assert mask.shape == (2, 1, 1, 2)
    assert mask[0, 0, 0, 0].item() is True
    assert mask[0, 0, 0, 1].item() is False  # padded token masked out!
    assert mask[1, 0, 0, 0].item() is True
    assert mask[1, 0, 0, 1].item() is True

