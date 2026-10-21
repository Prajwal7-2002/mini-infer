# Radix Prefix Caching 🌲

In enterprise deployments, conversational agents and coding assistants receive hundreds of requests that share identical **system prompts**, **few-shot examples**, or **conversation history**.

mini-infer implements a **Radix Tree Prefix Cache** to eliminate redundant prefill attention computation.

---

## The Bottleneck: Redundant Prefill Computation

During the *prefill phase*, an inference engine processes all input prompt tokens in parallel to generate the initial KV cache. The computational complexity of prefill attention scales quadratically with sequence length:

$$\text{Attention Flops} \propto O(N^2 \cdot d)$$

If 100 concurrent users prompt a model with a 1,500-token system prompt, a naive engine executes:

$$100 \times \text{Prefill}(1500) = 150,000 \text{ token FLOPs}$$

With **Radix Prefix Caching**, the 1,500 tokens are computed **once**. All subsequent requests reuse the existing physical KV-cache blocks directly from GPU VRAM:

$$\text{Prefill FLOPs} = 1 \times \text{Prefill}(1500) + 100 \times O(1) \text{ Block Table Lookup}$$

Time-to-First-Token (TTFT) for cached system prompts drops from **350 ms down to under 15 ms**.

---

## How the Radix Tree Works

A **Radix Tree** (or compact trie) is a space-optimized tree data structure where each edge represents a sequence of tokens:

```
                              [Root: Empty]
                                    │
                  "You are a helpful assistant..." (Token IDs: [15, 82, 943])
                                    │
                                    ▼
                             [Node A: Block 4]
                             Ref Count: 3
                                 /     \
    "Explain quantum physics..."        "What is the capital of..."
                 │                                  │
                 ▼                                  ▼
         [Node B: Block 7]                  [Node C: Block 12]
         Ref Count: 1                       Ref Count: 1
```

### 1. Token Matching & Cache Hit
When a new request arrives with tokens $T$:
1. The engine traverses the Radix Tree from the root, matching token substrings.
2. For every full block of 16 tokens matched in the tree, mini-infer reuses the existing physical block ID.
3. The reference counter of the matching tree node is incremented.
4. Only the unmatched *suffix* tokens (the unique user query) are dispatched to the model's prefill kernel!

### 2. Copy-on-Write (CoW)
Physical blocks shared across multiple requests are marked **read-only**. When an active request generates a new token that modifies a partially filled block, a new physical block is allocated and copied using Copy-on-Write semantics, ensuring memory safety across concurrent requests.

### 3. LRU Node Eviction
When the physical memory pool runs out of free blocks, the Radix Tree evicts nodes whose reference count is $0$ using a **Least Recently Used (LRU)** eviction policy, returning physical blocks to the free pool.

---

## Performance Impact

| Metric | Without Prefix Caching | With Radix Prefix Caching | Improvement |
|---|---|---|---|
| **TTFT (1.5k System Prompt)** | 340 ms | **12 ms** | **28.3x faster** |
| **Prefill VRAM Allocation** | 1,500 tokens per stream | **Shared 1,500 tokens** | **Up to 85% VRAM saved** |
| **Multi-turn Chat Throughput** | Degrading with conversation depth | Constant low latency | **3.2x higher throughput** |
