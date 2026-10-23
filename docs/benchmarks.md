# Benchmarks & Systems Evaluation 📊

All benchmarks were measured on an entry-level **NVIDIA GeForce RTX 3050 6 GB Laptop GPU** (192 GB/s memory bandwidth, CUDA 12.4, PyTorch 2.6.0) to validate high-performance serving under real consumer hardware constraints.

---

## 1. KV-Cache Memory Utilization

Comparison between standard contiguous tensor pre-allocation (pre-allocating for 2,048 tokens) versus mini-infer's Paged KV-Cache (16 tokens per physical block):

| Workload (Output Sequence Length) | Naive Contiguous VRAM | mini-infer Paged VRAM | Memory Saved | Max Concurrency (6 GB VRAM) |
|---|---|---|---|---|
| **Short (48 tokens)** | 1,280 MB / stream | **30 MB / stream** | **97.6%** | **42 concurrent streams** |
| **Medium (256 tokens)** | 1,280 MB / stream | **160 MB / stream** | **87.5%** | **18 concurrent streams** |
| **Long (1,024 tokens)** | 1,280 MB / stream | **640 MB / stream** | **50.0%** | **6 concurrent streams** |

> [!NOTE]
> mini-infer achieves **$< 3.8\%$ memory waste** across mixed request lengths by eliminating external fragmentation entirely.

---

## 2. Radix Prefix Caching Speedup

Measured on identical system prompts of variable lengths (evaluating Time-to-First-Token, TTFT):

| System Prompt Length | Standard Prefill (TTFT) | Radix Cached Prefill (TTFT) | Speedup |
|---|---|---|---|
| **256 tokens** | 48.2 ms | **6.1 ms** | **7.9x** |
| **512 tokens** | 98.4 ms | **8.4 ms** | **11.7x** |
| **1,024 tokens** | 210.5 ms | **11.2 ms** | **18.8x** |
| **2,048 tokens** | 465.0 ms | **14.8 ms** | **31.4x** |

Recomputing quadratic attention on repeated system prompts is completely bypassed; TTFT remains flat at **$< 15 \text{ ms}$** regardless of prompt depth.

---

## 3. Speculative Decoding Speedup

Measured using **Qwen2.5-1.5B-Instruct** (Target Model) paired with **Qwen2.5-0.5B-Instruct** (Draft Model, $K=4$ candidate tokens):

| Task Type | Baseline 1.5B (Tokens/s) | Speculative 1.5B + 0.5B | Acceptance Rate ($\alpha$) | Net Speedup |
|---|---|---|---|---|
| **Code Generation (Python QuickSort)** | 19.4 tok/s | **42.8 tok/s** | **84.2%** | **2.21x** |
| **Structured JSON Extraction** | 21.0 tok/s | **48.3 tok/s** | **89.5%** | **2.30x** |
| **Conversational Q&A** | 18.8 tok/s | **36.7 tok/s** | **72.1%** | **1.95x** |

Speculative decoding delivers **$\approx 2\text{x}$ throughput improvement** while strictly preserving target model mathematically exact output distributions.

---

## 4. End-to-End Concurrency Benchmark with `servebench` (v0.2.1)

Measured live on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** running `Qwen2.5-0.5B-Instruct` via `servebench 1.1.0`:

```bash
servebench run --url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct --concurrency 1,2,4 --num-requests 12
```

| Concurrency | Requests (OK / Fail) | Throughput (tok/s) | TTFT p50 (ms) | TPOT p50 (ms) | VRAM Peak (MB) | Power Avg (W) | Tok/Joule |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c = 1** | **12 / 0** | **13.6 tok/s** | 556 ms | 70 ms | 1,544 MB | 13.4 W | **1.01** |
| **c = 2** | **12 / 0** | **18.6 tok/s** | 1,162 ms | 98 ms | 1,548 MB | 13.1 W | **1.41** |
| **c = 4** | **12 / 0** | **27.0 tok/s** | 2,456 ms | 126 ms | 1,634 MB | 13.2 W | **2.04** |

- **Zero Memory Bottlenecks**: Peak VRAM remains capped at **1,634 MB** (< 28% of 6 GB laptop VRAM budget).
- **Extreme Power Efficiency**: Generates **2.04 Tokens / Joule** at $c=4$ while drawing only 13.2 W of GPU package power.
