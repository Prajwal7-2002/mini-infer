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
 
## 2026-10-23: Milestone 7 - Batched Paged Attention, Chunked Prefill & Preemption (v0.2.0)

- **Batched Paged KV Gather**: Wired physical `BlockPool` key/value memory tensors directly into `Qwen2Attention` with zero-copy block table gather (`gather_batched_paged_kv`) and per-token slot mapping.
- **Unified Batched Decoding**: Eliminated the serial unbatched decoding loop in `InferenceEngine.step()`. All concurrent active decoding streams now execute in a single batched forward pass (`[batch_size, 1]`), reducing DRAM weight transfers by up to 4×.
- **Chunked Prefill**: Implemented bounded prompt prefill slicing (`chunked_prefill_size=512`), interleaving prompt evaluation with active decodes to prevent ITL stutter during bursty request arrivals.
- **Memory Pressure Preemption**: Implemented graceful sequence preemption in `ContinuousScheduler`. When the physical block pool exhausts under load, the engine preempts the newest sequence, frees its blocks to let older requests complete, and resumes it without crashing.
- **Expanded Test Matrix**: Added `tests/test_engine.py` and test cases for chunked prefill, preemption, and paged read/write. 18/18 tests passing.

## 2026-11-05: v0.2.8 - Prefix cache correctness fix and honest TTFT numbers

An independent re-run found that a prefix-cache hit produced wrong output: with the cache on, Qwen2.5-0.5B
matched the Hugging Face reference for 0/40 tokens and ignored the system prompt.

- **Root causes** (`engine.py`): on a hit the suffix was prefilled with `past_key_values=None`, so it never
  attended to the cached prefix and its RoPE positions restarted at 0; then `write_sequence_kv(start_token_idx=0)`
  wrote the suffix KV over the shared cached blocks, corrupting later hits.
- **Fix**: load the cached blocks' KV as attention history; write only the new chunk at its real offset;
  if the whole prompt is cached, release the last block and recompute it privately; drop the contiguous
  KV copy after prefill (it duplicated the paged cache for the whole request).
- **Speed**: `BlockPool.write_sequence_kv` now does one indexed scatter per layer instead of a Python loop per
  token. Cold TTFT for an ~880-token prompt went from 2,264 ms to ~270 ms (measured in the same session).
- **Tests**: `tests/test_prefix_cache_correctness.py` compares the paged KV after a hit with a full forward
  pass, layer by layer (bf16/int8 KV, chunked and unchunked, fully cached prompt, re-run after other hits).
  It fails 4/5 on the old code. The earlier tests passed because a tiny random model just repeats the last
  token, so they could not see the bug.
- **Benchmark** (`scripts/bench_prefix_cache.py`, warmed up, median of 5, RTX 3050, Qwen2.5-0.5B bf16):

  | Prompt | No prefix cache | Cache hit | Speedup |
  | :--- | ---: | ---: | ---: |
  | ~935 tokens | 143 ms | 66 ms | 2.2× |
  | ~3,040 tokens | 480 ms | 68 ms | 7.0× |

  fp32: identical greedy output in 5/5 runs; mini-infer fp32 logits equal Hugging Face's exactly (max diff 0.0).
- **README**: removed the "29.7×", "38-70×", "< 15 ms" and "up to 80%" claims. The 29.7× compared an
  un-warmed first request (CUDA start-up) with a warm second one, on the broken cache. `test_real_repo.py`
  now warms up first.


## 2026-11-05: v0.2.9 - Prefix-cache eviction under memory pressure

- **Bug**: `PrefixCache.evict_lru` existed but nothing called it. Every finished prompt's blocks stayed cached,
  so a long-running server eventually raised `MemoryError: Paged KV-Cache pool exhausted` and dropped requests
  (14 of 24 failed at 4 concurrent users after ~70 requests in a servebench sweep), with only ~100 of 1,024
  blocks actually in use.
- **Fix**: `BlockPool.ensure_free(n)` asks a reclaimer for missing blocks before failing; the engine wires it to
  LRU eviction. Admission, prefill slot allocation and the decode pre-check all go through it. Eviction skips
  blocks a running request still holds (dropping the cache's reference would free nothing).
- **Tests**: 40 distinct prompts through a 32-block pool (failed on the old code); eviction never frees in-use
  blocks; the LRU unit test now checks both cases.
- **Benchmark** (servebench, RTX 3050 laptop, 0.5B bf16): 24/24 requests at c=1/2/4; 8.5 / 15.3 / 26.4 tok/s.
  The GPU is ~32% busy at ~810 MHz: mini-infer is CPU-bound (Python dispatch of per-layer paged-KV gathers),
  so results vary with the laptop's power state (14.9 tok/s at c=1 in an earlier run).

## 2026-11-05: v0.2.10 - Server reports the model it actually serves

- `/v1/models` was hard-coded to `qwen2.5-0.5b` / `Qwen/Qwen2.5-0.5B-Instruct`, so a server started with
  `--model Qwen/Qwen2.5-1.5B-Instruct` still claimed to serve 0.5B. Benchmark clients (servebench) record
  this, so results could be filed under the wrong model. It now lists the served model and its short alias
  (`qwen2.5-1.5b`).
- `/health` now reports `version` and `model` (servebench showed the version as unknown).
- Tests: `tests/test_server_endpoints.py` (fails on the old code). 51 tests.
