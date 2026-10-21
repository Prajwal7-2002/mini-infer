# Paged KV-Cache Memory Management 🧠

The Key-Value (KV) cache is the single largest consumer of GPU VRAM in large language model inference. mini-infer implements **PagedAttention-style block memory management** to eliminate GPU memory fragmentation.

---

## The Problem: Memory Fragmentation in Naive Systems

In standard Hugging Face or naive PyTorch implementations, tensors must be stored in **contiguous** virtual and physical memory addresses.

During autoregressive generation, the output sequence length is unknown ahead of time:

```
[Request 1: Max 2048 Tokens]  ──► Pre-allocates 2048 slots contiguous VRAM
                                 Actual generation: 42 tokens
                                 Wasted memory: (2048 - 42) / 2048 = 97.9% !
```

This creates two severe memory bottlenecks:

1. **Internal Fragmentation**: Pre-allocating contiguous memory for the theoretical maximum generation length (`max_new_tokens`) when most requests terminate early via EOS.
2. **External Fragmentation**: Dynamic requests of variable lengths allocating and freeing memory randomly, creating holes in GPU VRAM that prevent new requests from batching even when total free memory is ample.

In practice, naive systems waste between **60% and 80% of total GPU memory**, limiting concurrency to 2–4 parallel streams.

---

## The Solution: Operating System Paging for GPU Tensors

mini-infer borrows the fundamental concept of **virtual memory paging** from operating systems:

- **Physical Blocks**: VRAM is pre-allocated as a single fixed pool of blocks upon engine startup. Each block stores exactly $B = 16$ tokens for Key and Value projections across all attention layers.
- **Logical Blocks**: A client request sees its sequence as a contiguous sequence of logical blocks ($0, 1, 2, \dots$).
- **Block Table**: A dynamic mapping structure that maps a request's logical blocks to non-contiguous physical block IDs in GPU VRAM.

```
Request 1 (Logical)           Block Table              GPU Physical Memory Pool
┌─────────────┐             ┌─────────────┐            ┌────────────────────────┐
│ Logical 0   ├────────────►│ Logical 0: 4├───────────►│ Physical Block 4 (VRAM)│
├─────────────┤             ├─────────────┤            ├────────────────────────┤
│ Logical 1   ├────────────►│ Logical 1: 1├───────────►│ Physical Block 1 (VRAM)│
├─────────────┤             ├─────────────┤            ├────────────────────────┤
│ Logical 2   ├────────────►│ Logical 2: 7├───────────►│ Physical Block 7 (VRAM)│
└─────────────┘             └─────────────┘            └────────────────────────┘
```

---

## Memory Waste Analysis

With a block size of $B = 16$ tokens:
- **Zero External Fragmentation**: Because all physical blocks are identically sized ($16 \times \text{layers} \times \text{heads} \times d_k$), any free block can serve any request at any layer.
- **Bounded Internal Fragmentation**: Memory waste only exists in the *final* block of an active sequence:
  $$\text{Waste}_{\text{max}} = \frac{B - 1}{N} \le \frac{15}{N}$$
  For a typical request of 400 tokens, average memory waste drops to **$< 3.8\%$**, compared to $> 60\%$ in naive contiguous buffers.

---

## Python Implementation: Physical Block Manager

In mini-infer (`src/mini_infer/cache/memory_manager.py`), the physical pool is tracked using a fast stack of free block IDs:

```python
class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int = 16, num_layers: int = 24, num_kv_heads: int = 2, head_dim: int = 64, device: str = "cuda"):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        
        # Pre-allocated continuous VRAM tensor: [num_blocks, 2 (K/V), num_layers, num_kv_heads, block_size, head_dim]
        self.gpu_pool = torch.zeros(
            (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim),
            dtype=torch.float16,
            device=device
        )

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of KV-Cache Physical Blocks")
        return self.free_blocks.pop()

    def free_block(self, block_id: int):
        self.free_blocks.append(block_id)
```

During attention computation, keys and values are gathered dynamically using block table indices, avoiding expensive memory copies.
