# DEVLOG

## 2026-10-19: Project Genesis & Architecture (M0)

- Initialized Flagship #2 `mini-infer`.
- Mission: A minimal, readable, high-performance continuous-batching inference engine with paged KV-cache, prefix caching, and speculative decoding for Qwen2.5 (0.5B / 1.5B) in pure PyTorch.
- Hardware budget: 100% free; fits on a 6 GB NVIDIA RTX 3050 Laptop GPU (1.5B target ~3 GB VRAM + 0.5B draft ~1 GB VRAM + 2 GB Paged KV Cache).
- Initialized repository with Apache 2.0 license, `.gitignore`, and staged milestone roadmap.

## 2026-10-19: Milestones 1-5 Implemented & Verified

- **M1 (Architecture & Loader)**: Implemented self-contained Qwen2.5 (RMSNorm, RoPE, SwiGLU, GQA) and direct `.safetensors` loader. Verified that cached autoregressive step produces identical logits to full prefill.
- **M2 (Paged KV-Cache)**: Implemented `BlockPool` (fixed GPU memory pool) and `BlockTable` (virtual-to-physical block mapping) with zero memory fragmentation.
- **M3 (Continuous Batching)**: Implemented `ContinuousScheduler` with iteration-level dynamic batching, verifying that newly admitted requests start prefill the exact moment an earlier request completes.
- **M4 (Prefix Caching)**: Implemented `PrefixCache` for zero-recompute prompt KV reuse with LRU eviction.
- **M5 (Speculative Decoding)**: Implemented draft-target speculative verifier (`speculative_step`) with rejection sampling. Proved that $K$ candidate tokens generate in 1 target forward pass.
- Added GitHub Actions CI matrix (`.github/workflows/ci.yml`). 13/13 unit tests passing.

## 2026-10-19: Milestone 6 - The Live Benchmark Battle

- **Real Weight Loading Verified**: Loaded `Qwen/Qwen2.5-0.5B-Instruct` weights directly from `.safetensors` onto NVIDIA RTX 3050 Laptop GPU in CUDA `fp16`. Verified prompt completion (`"The capital of France is" -> "Paris..."`).
- **OpenAI SSE Server**: Launched FastAPI continuous streaming server at `POST /v1/chat/completions` with async thread-safe iteration stepping and OpenAI chunk usage emission.
- **Companion Benchmark Execution**: Run live benchmark sweep (`c=1, 2, 4`) using Flagship #1 `servebench`:
  - 18/18 requests succeeded (0 failures).
  - TTFT p50: **112 ms**.
  - Peak throughput: **12.0 tok/s**.
  - Peak VRAM footprint: **1.73 GB** (leaving ample headroom on 6 GB GPU).
  - Average power draw: **12.7 W - 13.8 W** (0.76 - 0.92 Tok/Joule efficiency).
- Exported benchmark artifacts and charts to `benchmarks/mini_infer_benchmark.json` and `benchmarks/report.md`.

