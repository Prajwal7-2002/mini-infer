import torch


class BlockPool:
    """Fixed-size GPU memory pool for Paged KV-Cache blocks.

    Instead of allocating a large contiguous buffer per request, the engine pre-allocates
    a single fixed pool of blocks on the GPU and maps requests dynamically via a block table.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Free list of physical block indices
        self.free_list: list[int] = list(range(num_blocks))

        # Reference count per block for prefix caching / copy-on-write
        self.ref_counts: dict[int, int] = {i: 0 for i in range(num_blocks)}

        # Pre-allocate physical KV tensors per layer
        # Shape per layer: [num_blocks, num_kv_heads, block_size, head_dim]
        self.key_caches: list[torch.Tensor] = [
            torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.value_caches: list[torch.Tensor] = [
            torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    @property
    def num_free(self) -> int:
        """Number of currently available unallocated physical blocks."""
        return len(self.free_list)

    def allocate_block(self) -> int:
        """Allocate a single physical block from the free list."""
        if not self.free_list:
            raise MemoryError("Paged KV-Cache pool exhausted! Out of GPU memory blocks.")
        block_id = self.free_list.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def allocate_blocks(self, count: int) -> list[int]:
        """Allocate multiple physical blocks."""
        if len(self.free_list) < count:
            raise MemoryError(
                f"Cannot allocate {count} blocks; only {len(self.free_list)} available."
            )
        allocated = []
        for _ in range(count):
            allocated.append(self.allocate_block())
        return allocated

    def retain_block(self, block_id: int):
        """Increment reference count for a block (used in prefix caching)."""
        self.ref_counts[block_id] += 1

    def free_block(self, block_id: int):
        """Decrement reference count; return to free list if count reaches 0."""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] <= 0:
            self.ref_counts[block_id] = 0
            self.free_list.append(block_id)

    def free_blocks(self, block_ids: list[int]):
        """Free a list of blocks."""
        for b_id in block_ids:
            self.free_block(b_id)


class BlockTable:
    """Logical-to-physical block mapping for an active request sequence."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.physical_block_ids: list[int] = []
        self.num_tokens: int = 0

    def append_slot(self, block_pool: BlockPool) -> tuple[int, int]:
        """Reserve the next token slot.

        Allocates a new physical block from the pool if the current block is full.
        Returns:
            Tuple of (physical_block_id, slot_offset_within_block).
        """
        slot_offset = self.num_tokens % self.block_size
        if slot_offset == 0:
            # Need a new physical block
            new_block_id = block_pool.allocate_block()
            self.physical_block_ids.append(new_block_id)

        current_block_id = self.physical_block_ids[-1]
        self.num_tokens += 1
        return current_block_id, slot_offset

    def free(self, block_pool: BlockPool):
        """Free all physical blocks associated with this request."""
        block_pool.free_blocks(self.physical_block_ids)
        self.physical_block_ids.clear()
        self.num_tokens = 0
