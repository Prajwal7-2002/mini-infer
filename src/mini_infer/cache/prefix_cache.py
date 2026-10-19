import time

from mini_infer.cache.block_pool import BlockPool


class PrefixCache:
    """Radix / Hash-based Prefix Cache for zero-recompute prompt KV reuse.

    Stores and matches chains of fixed-size token blocks so that common prefixes
    (system instructions, multi-turn chat history, few-shot examples) bypass prefill
    and reuse pre-computed KV tensors directly from GPU VRAM.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        # Maps chained block tuple hash -> physical_block_id
        self.cache: dict[tuple[int, ...], int] = {}
        # Tracks last access time for LRU eviction: physical_block_id -> timestamp
        self.last_accessed: dict[int, float] = {}

    def match_prefix(
        self, token_ids: list[int], block_pool: BlockPool
    ) -> tuple[list[int], int]:
        """Find the longest cached block sequence that matches the start of token_ids.

        Args:
            token_ids: Full prompt token sequence.
            block_pool: Global block pool to retain matched blocks.

        Returns:
            Tuple of (matched_physical_block_ids, num_cached_tokens).
        """
        matched_blocks = []
        num_full_blocks = len(token_ids) // self.block_size

        prefix_tokens: list[int] = []
        for i in range(num_full_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            prefix_tokens.extend(chunk)
            key = tuple(prefix_tokens)

            if key in self.cache:
                physical_block_id = self.cache[key]
                # Retain in memory pool
                block_pool.retain_block(physical_block_id)
                matched_blocks.append(physical_block_id)
                self.last_accessed[physical_block_id] = time.time()
            else:
                break

        num_cached_tokens = len(matched_blocks) * self.block_size
        return matched_blocks, num_cached_tokens

    def insert_prefix_block(
        self,
        full_prefix_tokens_up_to_block: tuple[int, ...],
        physical_block_id: int,
        block_pool: BlockPool,
    ):
        """Register a newly computed full block into the prefix cache."""
        assert len(full_prefix_tokens_up_to_block) % self.block_size == 0
        key = tuple(full_prefix_tokens_up_to_block)

        if key not in self.cache:
            self.cache[key] = physical_block_id
            block_pool.retain_block(physical_block_id)
            self.last_accessed[physical_block_id] = time.time()

    def evict_lru(self, block_pool: BlockPool, count: int = 1) -> int:
        """Evict the least recently used cached prefix blocks when VRAM is tight."""
        evicted = 0
        # Sort by access timestamp ascending
        sorted_blocks = sorted(self.last_accessed.items(), key=lambda x: x[1])

        for block_id, _ in sorted_blocks:
            if evicted >= count:
                break
            # Find the cache key
            keys_to_delete = [k for k, v in self.cache.items() if v == block_id]
            for k in keys_to_delete:
                del self.cache[k]
            del self.last_accessed[block_id]
            block_pool.free_block(block_id)
            evicted += 1

        return evicted
