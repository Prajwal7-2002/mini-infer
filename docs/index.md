# mini-infer 🚀

[![PyPI version](https://img.shields.io/pypi/v/mini-infer.svg)](https://pypi.org/project/mini-infer/)
[![CI](https://github.com/Prajwal7-2002/mini-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/mini-infer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Prajwal7-2002/mini-infer/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)

A minimal, educational, and high-performance **continuous-batching LLM inference engine** with **Paged KV-cache**, **radix prefix caching**, and **speculative decoding** built from scratch in pure Python and PyTorch.

---

## Why mini-infer?

Most practitioners know how to *call* an inference engine (`ollama run` or `vllm serve`). However, the underlying systems engineering that powers modern production LLM serving:

1. **How tensors are paged in GPU VRAM** to eliminate external memory fragmentation.
2. **How continuous batching dynamically schedules requests** at the token iteration level rather than request level.
3. **How prefix caching avoids redundant prefills** across shared system prompts via radix trees.
4. **How speculative decoding accelerates generation** by interleaving small draft proposals with target model parallel verification.

...are typically locked behind tens of thousands of lines of complex CUDA, C++, and distributed runtime code.

**mini-infer** implements these exact production algorithms in **pure, readable, testable Python/PyTorch**. It runs on consumer hardware (such as an entry-level 6 GB laptop GPU or a free Kaggle T4), with **zero C++ compiler or CMake toolchain required**.

---

## Systems Architecture

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

## Core Capabilities at a Glance

| Component | Industry Baseline | mini-infer Implementation |
|---|---|---|
| **Memory Allocation** | Static pre-allocated contiguous tensors (60–80% VRAM waste) | **PagedAttention Block Pool**: 16 tokens/block with virtual block tables (< 4% memory waste) |
| **Batched Paged Decode (v0.2.0)** | Serial loop per sequence or custom CUDA kernel | **Unified `[B, 1]` Paged Forward Pass**: Direct physical tensor gather with zero-overhead slot mapping |
| **Prompt Ingestion (v0.2.0)** | Unbounded monolithic prefill starving decode streams | **Chunked Prefill**: Bounded 512-token chunks interleaving prompt ingestion with active decodes |
| **OOM Resilience (v0.2.0)** | Unhandled `CUDA out of memory` process crashes | **Graceful Preemption**: Re-queues lowest-priority sequences back to `WAITING` when pool exhausts |
| **Prompt Caching** | Recomputing $O(N^2)$ attention on identical system prompts | **Radix Tree Prefix Cache**: Zero recompute on shared conversation prefixes |
| **Batching Strategy** | Naive request batching (waiting for slowest request to finish) | **Continuous Iteration Batching**: Dynamic token-level scheduling |
| **Generation Speed** | Sequential auto-regressive generation ($1$ token per step) | **Speculative Decoding**: $K$-token draft proposals with parallel target verification |
| **Model Architecture** | Black-box weights | **Pure Qwen2.5 safetensors loader**: Custom RMSNorm, RoPE, and SDPA |

---

## Design Principles

- **Zero Build Toolchain**: No Visual Studio Build Tools, gcc, or nvcc required. Standard PyTorch CUDA wheels power all tensor kernels.
- **Strictly Typed & Verified**: Full typing with comprehensive unit tests (`pytest`) covering KV-cache paging, Radix prefix matching, and speculative verification math.
- **Consumer Hardware Native**: Runs comfortably within 6 GB VRAM budgets (RTX 3050, GTX 1660 Ti) while scaling up to cloud GPUs (A10, T4, A100).
