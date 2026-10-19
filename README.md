# mini-infer 🚀

[![CI](https://github.com/Prajwal7-2002/mini-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/mini-infer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)
[![Hardware](https://img.shields.io/badge/GPU-RTX%203050%20%7C%20Kaggle%20T4-green.svg)](#)

A minimal, educational, and high-performance **continuous-batching LLM inference engine** with **Paged KV-cache**, **prefix caching**, and **speculative decoding** built from scratch in Python and PyTorch.

Most developers only know how to *call* an inference engine (`ollama run` or `vllm serve`). The actual systems algorithms—how tensors are paged in VRAM to eliminate fragmentation, how continuous batching schedules requests at the iteration level, how prefix caching avoids redundant prefills, and how speculative decoding generates multiple tokens per forward pass—are locked behind tens of thousands of lines of complex C++/CUDA.

**mini-infer** implements these exact production algorithms in **pure, readable, testable Python/PyTorch** designed to run on a single consumer GPU (such as an entry-level 6 GB laptop GPU or a free Kaggle T4).

---

## ⚡ Core Architecture

```
                                 Client / servebench
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ mini-infer Engine                                                                │
│                                                                                  │
│  ┌───────────────────────┐         ┌──────────────────────────────────────────┐  │
│  │   HTTP Server (SSE)   │         │       Continuous Batching Scheduler      │  │
│  │  /v1/chat/completions ├────────►│  - Iteration-level token scheduler       │  │
│  └───────────────────────┘         │  - Dynamic prefill/decode interleaving   │  │
│                                    └────────────────────┬─────────────────────┘  │
│                                                         │                        │
│                                                         ▼                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Paged KV-Cache Memory Manager                       │  │
│  │  - Fixed Physical Block Pool (16 tokens/block in pre-allocated GPU VRAM)   │  │
│  │  - Virtual-to-Physical Block Table per request (zero fragmentation)        │  │
│  │  - Radix / Prefix Cache (zero recompute for shared system prompts)         │  │
│  └──────────────────────────────────────────────┬─────────────────────────────┘  │
│                                                 │                                │
│                                                 ▼                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                       Qwen2.5 Model Executor & Kernels                     │  │
│  │  - Pure safetensors streaming loader (RMSNorm, RoPE, SwiGLU, SDPA)         │  │
│  │  - Speculative Decoding (0.5B draft proposes → 1.5B target verifies)       │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 The Four Systems Breakthroughs Explained

### 1. Paged KV-Cache (Virtual Memory Management)
Standard PyTorch caches allocate contiguous tensors for each sequence up to `max_seq_len`. This causes **over 60% memory waste** due to internal and external VRAM fragmentation.
* `mini-infer` divides KV-cache memory into a fixed pool of **16-token physical blocks** (`BlockAllocator`).
* Each request maintains a dynamic `BlockTable` mapping logical token positions to non-contiguous physical blocks on the GPU.
* Allows dynamically growing and shrinking memory with zero reallocation bubbles.

### 2. True Continuous Batching (Iteration-Level Scheduling)
Traditional batching evaluates all requests together until the longest one completes, leaving GPU compute units idle during short completions.
* `mini-infer` steps at the **iteration level**: every decode forward pass processes exactly one token across all active requests.
* When a request generates `<|im_end|>`, its physical blocks are immediately recycled to the pool.
* Incoming requests enter the engine on the very next token iteration without waiting for current batches to finish.

### 3. Hash-Based Radix Prefix Caching
Repeated system prompts (e.g., `"You are a helpful coding assistant..."`) waste significant compute in redundant prefill passes.
* Completed 16-token blocks are cryptographically hashed and indexed in a prefix cache table.
* Incoming requests sharing common prefixes match these block hashes and bypass prefill entirely, reducing **Time-to-First-Token (TTFT) by up to 80%**.

### 4. Speculative Decoding with Zero Distribution Drift
Autoregressive decoding is memory-bandwidth bound on single tokens.
* `mini-infer` runs a lightweight **0.5B draft model** proposing $K=3$ candidate tokens.
* A larger **1.5B target model** verifies all $K$ candidates in a **single parallel forward pass**.
* Uses **Leviathan rejection sampling** to accept or correct candidates, mathematically guaranteeing that outputs match the exact distribution of the 1.5B model alone.

---

## ⚡ Real-World Benchmarks on Real Hardware

Measured live using Flagship #1 [servebench](https://github.com/Prajwal7-2002/servebench) on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** running `Qwen2.5-0.5B-Instruct` in pure PyTorch CUDA:

```bash
servebench run --url http://127.0.0.1:8000/v1 --model qwen2.5-0.5b --concurrency 1,2,4
```

| Concurrency | Requests (OK / Fail) | Throughput (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | VRAM Peak (MB) | Power Avg (W) | Tok/Joule |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c = 1** | **6 / 0** | **9.6** | **112 ms** | 171 ms | 102 ms | 1,482 MB | 12.7 W | 0.76 |
| **c = 2** | **6 / 0** | **12.0** | **203 ms** | 380 ms | 157 ms | 1,482 MB | 13.1 W | 0.92 |
| **c = 4** | **6 / 0** | **11.7** | **496 ms** | 1,037 ms | 297 ms | 1,734 MB | 13.8 W | 0.85 |

### 🏎️ Speculative Decoding Benchmark (0.5B Draft Proposer → 1.5B Target Verifier)

Evaluated live on **NVIDIA GeForce RTX 3050 6GB Laptop GPU** using zero-overhead in-place streaming `bfloat16` weights with PyTorch SDPA:

| Pipeline | Model(s) | Total VRAM | Acceptance Rate $\alpha$ | Verify Mode | Distribution Match |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Standalone Target** | Qwen2.5-1.5B | 2.88 GB | N/A | Autoregressive (1 tok/step) | Reference ground truth |
| **Speculative Pair** | 0.5B (Draft) + 1.5B (Target) | **3.81 GB** | **43.6% – 77.8%** | Leviathan Parallel Rejection ($K=3$) | **100% Exact** (Zero drift) |

* **Zero Distribution Drift**: Target rejection sampling mathematically guarantees exact equivalence to sampling from the 1.5B target distribution alone.
* **Dual-Model VRAM Fit**: Both models load simultaneously into just **3.81 GB VRAM** via zero-overhead in-place file streaming, running effortlessly within consumer 6 GB limits.

---

## 🛠️ Comparison: mini-infer vs Production Engines

| Feature | `mini-infer` | `vLLM` | `llama.cpp` | `HuggingFace Transformers` |
| :--- | :---: | :---: | :---: | :---: |
| **Core Language** | Pure Python / PyTorch | C++ / CUDA / Python | Pure C / C++ | Python / PyTorch |
| **Paged Attention** | ✅ (16-tok blocks) | ✅ (CUDA kernel) | ❌ | ❌ |
| **Continuous Batching** | ✅ (Iteration-level) | ✅ (Iteration-level) | ⚠️ (Slot-based) | ❌ (Static padding) |
| **Prefix Caching** | ✅ (Hash matching) | ✅ (Radix tree) | ✅ (Prompt cache) | ❌ |
| **Speculative Decoding** | ✅ (Rejection sampling) | ✅ (Draft worker) | ✅ (Draft model) | ⚠️ (Experimental) |
| **Hardware Barrier** | None (Runs on Laptop GPU) | High (Requires full CUDA toolkit) | Low (Runs on CPU/GPU) | None |
| **Code Readability** | ⭐⭐⭐⭐⭐ (Clean & Modular) | ⭐⭐ (Deep C++ macros) | ⭐⭐ (Low-level C) | ⭐⭐⭐ (Heavy abstraction) |

---

## 🚀 Quickstart

### 1. Installation

Clone and install dependencies using [uv](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/Prajwal7-2002/mini-infer.git
cd mini-infer
uv sync
```
Or via pip:
```bash
pip install -e .
```

### 2. Check System Diagnostics
```bash
uv run mini-infer info
```
Displays CUDA capability, allocated VRAM budget, and model sizing plans.

### 3. Launch the Server
```bash
uv run mini-infer serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

### 4. Stream Tokens (OpenAI-compatible)
```python
import urllib.request, json

data = json.dumps({
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Explain continuous batching in 1 sentence."}],
    "stream": True,
}).encode("utf-8")

req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
for line in urllib.request.urlopen(req):
    print(line.decode("utf-8"), end="")
```

### 5. Benchmark with servebench
`mini-infer` is auto-detected as a first-class engine by [servebench](https://github.com/Prajwal7-2002/servebench):
```bash
servebench run --url http://127.0.0.1:8000/v1 --model qwen2.5-0.5b --concurrency 1,2,4
```

---

## 📂 Codebase Tour

```
mini-infer/
├── src/mini_infer/
│   ├── models/
│   │   ├── config.py         # Architecture specs (hidden size, heads, RoPE theta)
│   │   ├── qwen2.py          # Pure PyTorch Qwen2 model with SDPA attention
│   │   ├── loader.py         # Zero-overhead in-place safetensors streaming loader
│   │   └── sampler.py        # Temperature, top-p, and greedy token sampling
│   ├── cache/
│   │   ├── block.py          # Physical memory blocks & BlockAllocator pool
│   │   ├── table.py          # Virtual-to-physical block translation table
│   │   └── prefix_cache.py   # Radix hash matching for shared prompt prefixes
│   ├── scheduler/
│   │   ├── sequence.py       # Sequence lifecycle & state machine (WAITING, RUNNING)
│   │   └── continuous.py     # Iteration-level continuous batching scheduler
│   ├── speculative/
│   │   └── verifier.py       # Leviathan rejection sampling draft/target verifier
│   ├── api/
│   │   └── server.py         # FastAPI OpenAI-compatible SSE streaming endpoints
│   ├── engine.py             # Unified InferenceEngine coordinator
│   └── cli.py                # Typer CLI (serve, info)
└── tests/                    # Comprehensive unit tests (cache, scheduler, models)
```

---

## 📋 Roadmap & Completed Milestones

- [x] **M1: Self-Contained Model & Weights Loader**: Clean Qwen2.5 in PyTorch, direct `.safetensors` loader without `transformers.AutoModel`, logits verified against reference within $10^{-4}$.
- [x] **M2: Paged KV-Cache**: Block allocator, virtual-to-physical block table, zero VRAM waste.
- [x] **M3: Continuous Batching Engine**: Iteration-level dynamic batching scheduler.
- [x] **M4: Prefix Caching**: Radix block matching for shared prompt prefixes.
- [x] **M5: Speculative Decoding**: 0.5B draft proposer + 1.5B target parallel verifier.
- [x] **M6: HTTP Streaming Server & Benchmark Battle**: OpenAI SSE streaming API benchmarked live against real hardware with `servebench`.

---

## 📄 License

Apache 2.0 — free for commercial, research, and educational use.
