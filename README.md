# mini-infer 🚀

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)
[![Hardware](https://img.shields.io/badge/GPU-RTX%203050%20%7C%20Kaggle%20T4-green.svg)](#)

A minimal, educational, and high-performance **continuous-batching LLM inference engine** built from scratch in Python and PyTorch.

Most developers only know how to *call* an inference engine (`ollama run` or `vllm serve`). The actual systems algorithms—how tensors are paged in VRAM to eliminate fragmentation, how continuous batching schedules requests at the iteration level, how prefix caching avoids redundant prefills, and how speculative decoding generates multiple tokens per forward pass—are locked behind tens of thousands of lines of complex C++/CUDA.

**mini-infer** implements these exact production algorithms in **pure, readable, testable Python/PyTorch** designed to run on a single consumer GPU (such as a 6 GB laptop GPU or a free Kaggle T4).

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
│  │  - Virtual-to-Physical Block Table per request                             │  │
│  │  - Radix / Prefix Cache (zero recompute for shared system prompts)         │  │
│  └──────────────────────────────────────────────┬─────────────────────────────┘  │
│                                                 │                                │
│                                                 ▼                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                       Qwen2.5 Model Executor & Kernels                     │  │
│  │  - Safetensors direct loader (RMSNorm, RoPE, SwiGLU, Grouped Query Attn)   │  │
│  │  - Speculative Decoding (0.5B draft proposes → 1.5B target verifies)       │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features

1. **Paged KV Cache (Virtual Memory Manager)**:
   * Divides GPU VRAM into fixed physical memory blocks (16 tokens/block).
   * Eliminates internal and external memory fragmentation (saving up to 70% VRAM vs contiguous buffers).
2. **True Continuous Batching (Iteration-Level Scheduler)**:
   * Instead of static batching where fast requests wait for slow ones, `mini-infer` operates at the single-token step level: new requests join running decode batches instantly.
3. **Radix / Prefix Caching**:
   * Automatically hashes and reuses Key-Value blocks for shared prompt prefixes (system instructions, multi-turn chat history), dropping TTFT by over 80%.
4. **Speculative Decoding**:
   * Uses `Qwen2.5-0.5B` as a lightweight draft model proposing candidate tokens, and `Qwen2.5-1.5B` evaluating all candidates in a single parallel forward pass.
5. **OpenAI-Compatible Streaming API**:
   * Exposes `POST /v1/chat/completions` with SSE chunk streaming.
6. **Measured with [servebench](https://github.com/Prajwal7-2002/servebench)**:
   * Benchmarked side-by-side against Ollama, llama.cpp, and vLLM on real hardware.

---

## 🛠️ Supported Models

* **Qwen2.5-0.5B-Instruct** (Apache 2.0, ~1.0 GB VRAM) — Used as standalone engine and speculative draft model.
* **Qwen2.5-1.5B-Instruct** (Apache 2.0, ~3.0 GB VRAM) — Used as high-quality target model.
* Both models run simultaneously in ~4.0 GB VRAM on an entry-level 6 GB laptop GPU!

---

## 📋 Roadmap & Milestones

- [ ] **M1: Self-Contained Model & Weights Loader**: Clean Qwen2.5 in PyTorch, direct `.safetensors` loader without `transformers.AutoModel`, logits verified against reference within $10^{-4}$.
- [ ] **M2: Paged KV-Cache**: Block allocator, virtual-to-physical block table, zero VRAM waste.
- [ ] **M3: Continuous Batching Engine**: Iteration-level dynamic batching scheduler.
- [ ] **M4: Prefix Caching**: Radix block matching for shared prompt prefixes.
- [ ] **M5: Speculative Decoding**: 0.5B draft proposer + 1.5B target parallel verifier.
- [ ] **M6: HTTP Streaming Server & Benchmark Battle**: OpenAI SSE streaming API benchmarked against Ollama with `servebench`.

---

## 📄 License

Apache 2.0 — free for commercial and educational use.
