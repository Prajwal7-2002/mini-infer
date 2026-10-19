import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.prefix_cache import PrefixCache


def test_prefix_cache_hit_and_miss():
    pool = BlockPool(
        num_blocks=10,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = PrefixCache(block_size=16)

    # 48 tokens = 3 full blocks
    system_prompt = list(range(100, 148))

    # Simulate prefill of request 1:
    b0 = pool.allocate_block()
    cache.insert_prefix_block(tuple(system_prompt[:16]), b0, pool)

    b1 = pool.allocate_block()
    cache.insert_prefix_block(tuple(system_prompt[:32]), b1, pool)

    b2 = pool.allocate_block()
    cache.insert_prefix_block(tuple(system_prompt[:48]), b2, pool)

    # Now Request 2 arrives with the same 48-token system prompt + a new 10-token question
    user_query = system_prompt + [500, 501, 502, 503, 504]

    matched_blocks, num_cached = cache.match_prefix(user_query, pool)
    assert matched_blocks == [b0, b1, b2]
    assert num_cached == 48

    # Request 3 arrives with an unrelated prompt
    unrelated_query = [999] * 32
    matched_unrelated, num_unrelated = cache.match_prefix(unrelated_query, pool)
    assert matched_unrelated == []
    assert num_unrelated == 0


def test_prefix_cache_lru_eviction():
    pool = BlockPool(
        num_blocks=4,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = PrefixCache(block_size=16)

    b0 = pool.allocate_block()
    cache.insert_prefix_block(tuple(range(16)), b0, pool)

    assert pool.num_free < 4
    # Evict block
    evicted = cache.evict_lru(pool, count=1)
    assert evicted == 1

    # Attempting to match now yields a miss
    matched, num_cached = cache.match_prefix(list(range(16)), pool)
    assert matched == []
    assert num_cached == 0
