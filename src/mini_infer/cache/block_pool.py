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

    def write_slot(
        self,
        layer_idx: int,
        block_id: int,
        slot_offset: int,
        k_slot: torch.Tensor,
        v_slot: torch.Tensor,
    ) -> None:
        """Write key and value states for a single token slot into physical block pool."""
        if k_slot.dim() == 3:
            k_slot = k_slot.squeeze(0)
            v_slot = v_slot.squeeze(0)
        self.key_caches[layer_idx][block_id, :, slot_offset, :] = k_slot
        self.value_caches[layer_idx][block_id, :, slot_offset, :] = v_slot

    def write_sequence_kv(
        self,
        layer_idx: int,
        physical_block_ids: list[int],
        start_token_idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
    ) -> None:
        """Write a sequence of key and value states into physical block pool."""
        if k_seq.dim() == 4:
            k_seq = k_seq.squeeze(0)
            v_seq = v_seq.squeeze(0)
        num_tokens = k_seq.shape[1]
        for i in range(num_tokens):
            token_pos = start_token_idx + i
            block_idx = token_pos // self.block_size
            slot_offset = token_pos % self.block_size
            block_id = physical_block_ids[block_idx]
            self.key_caches[layer_idx][block_id, :, slot_offset, :] = k_seq[:, i, :]
            self.value_caches[layer_idx][block_id, :, slot_offset, :] = v_seq[:, i, :]

    def read_sequence_kv(
        self,
        layer_idx: int,
        physical_block_ids: list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather contiguous [1, num_kv_heads, num_tokens, head_dim] tensors for a sequence."""
        if num_tokens <= 0 or not physical_block_ids:
            empty = torch.empty(
                (1, self.num_kv_heads, 0, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            return empty, empty

        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        needed_block_ids = physical_block_ids[:num_blocks_needed]

        # Extract blocks: [num_blocks_needed, num_kv_heads, block_size, head_dim]
        k_blocks = self.key_caches[layer_idx][needed_block_ids]
        v_blocks = self.value_caches[layer_idx][needed_block_ids]

        # Reshape to [num_kv_heads, total_slots, head_dim]
        n_b = len(needed_block_ids)
        k_contig = k_blocks.permute(1, 0, 2, 3).reshape(self.num_kv_heads, n_b * self.block_size, self.head_dim)
        v_contig = v_blocks.permute(1, 0, 2, 3).reshape(self.num_kv_heads, n_b * self.block_size, self.head_dim)

        # Slice to exact num_tokens and unsqueeze batch dimension: [1, num_kv_heads, num_tokens, head_dim]
        k_out = k_contig[:, :num_tokens, :].unsqueeze(0)
        v_out = v_contig[:, :num_tokens, :].unsqueeze(0)
        return k_out, v_out

    def gather_batched_paged_kv(
        self,
        layer_idx: int,
        block_tables: list[list[int]],
        context_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather KV tensors and attention mask for a batch of sequences.

        Returns:
            k_batch: [B, num_kv_heads, max_len, head_dim]
            v_batch: [B, num_kv_heads, max_len, head_dim]
            mask: [B, 1, 1, max_len] (bool mask where True indicates valid token)
        """
        B = len(block_tables)
        max_len = max(context_lens) if context_lens else 1

        k_batch = torch.zeros(
            (B, self.num_kv_heads, max_len, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_batch = torch.zeros(
            (B, self.num_kv_heads, max_len, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        mask = torch.zeros((B, 1, 1, max_len), device=self.device, dtype=torch.bool)

        for i in range(B):
            n_tok = context_lens[i]
            b_ids = block_tables[i]
            if n_tok > 0 and b_ids:
                k_seq, v_seq = self.read_sequence_kv(layer_idx, b_ids, n_tok)
                k_batch[i, :, :n_tok, :] = k_seq[0]
                v_batch[i, :, :n_tok, :] = v_seq[0]
                mask[i, 0, 0, :n_tok] = True

        return k_batch, v_batch, mask


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
