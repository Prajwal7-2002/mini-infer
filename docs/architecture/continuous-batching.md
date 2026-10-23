# Continuous Batching & Iteration Scheduling 🔄

Traditional deep learning systems batch requests at the **request level** (e.g., waiting for all $N$ requests in a batch to finish generating before returning responses). 

Because different requests generate wildly different numbers of tokens (e.g., Request A generates 10 tokens, Request B generates 500 tokens), static request batching leaves GPU cores idle waiting for the longest request:

```
Static Batching (Request-Level):
Request A: [Prefill][D1][D2][D3][EOS] ──────► [ IDLE / PADDED SLOTS ] ─────► Done
Request B: [Prefill][D1][D2][D3][D4] ... [D500][EOS] ─────────────────────► Done
```

mini-infer implements **Continuous (Iteration-Level) Batching**, where the batch is dynamically reassembled after *every single token forward pass*.

---

## Iteration-Level Scheduling

In mini-infer's scheduler (`src/mini_infer/engine/scheduler.py`):

1. **New Request Ingestion**: As soon as a request finishes generating an `<|im_end|>` token, its KV blocks are freed and returned to the memory pool.
2. **Immediate Slot Re-allocation**: A newly arrived pending request is scheduled into the very next forward pass, without waiting for other concurrent requests to complete.
3. **Interleaved Prefill & Decode**: The engine multiplexes compute-bound prefill iterations alongside memory-bound decode iterations.

```
Continuous Batching (Iteration-Level):
Step 1:  [Req A: Token 1]  [Req B: Token 40]  [Req C: Prefill Phase]
Step 2:  [Req A: Token 2]  [Req B: Token 41]  [Req C: Token 1]
Step 3:  [Req A: EOS] ──► Freed!
Step 4:  [Req D: Prefill]  [Req B: Token 42]  [Req C: Token 2]
```

---

## The Prefill vs. Decode Asymmetry

| Phase | Computational Characteristic | GPU Bottleneck | Arithmetic Intensity |
|---|---|---|---|
| **Prefill** | Processes all prompt tokens in parallel | **Tensor Compute (FLOPs)** | High ($> 100 \text{ FLOPs/byte}$) |
| **Decode** | Generates $1$ token sequentially per request | **Memory Bandwidth (GB/s)** | Low ($\approx 1 \text{ FLOP/byte}$) |

By co-scheduling requests in decode phase with an active prefill chunk, mini-infer saturates both the GPU's memory bus and its CUDA tensor cores simultaneously, maximizing overall token throughput (Goodput).

---

## Starvation Prevention: Chunked Prefill (v0.2.0)

If a single request submits an enormous prompt (e.g., a 10,000-token prompt), naive engines block all existing decode streams for several seconds, causing massive latency spikes (Inter-Token Latency degradation).

mini-infer `v0.2.0` solves this with **Chunked Prefill**:
- **Chunked Slicing**: Prompts are sliced into chunks of `chunked_prefill_size=512`.
- **Interleaving**: The scheduler ingests at most one 512-token chunk per step, interleaving prompt processing with active decode steps.
- **Bounded ITL**: Time-to-First-Token for the new request proceeds smoothly while active decode streams maintain uninterrupted, steady token emission.

---

## Memory Pressure Preemption (v0.2.0)

During burst traffic spikes, requests may demand more physical blocks than available in `BlockPool`. Without protection, this causes hard `CUDA out of memory` or `MemoryError` crashes.

In `mini-infer v0.2.0`, the continuous scheduler implements graceful **Preemption Recovery**:
1. When `block_pool.can_allocate()` fails for active decodes or prefills, the scheduler selects the lowest-priority (most recently scheduled) active sequence.
2. The sequence is suspended and re-queued to the `WAITING` state, and its physical blocks are freed back to the pool.
3. In-flight high-priority sequences continue uninterrupted to completion.
4. Once blocks are freed by retiring finished sequences, the preempted sequence is re-scheduled and resumed automatically.
