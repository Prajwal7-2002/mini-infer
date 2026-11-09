---
title: mini-infer - PagedAttention KV-Cache & Prefix Studio
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: Interactive GPU PagedAttention block pool & prefix cache.
tags:
  - pagedattention
  - kv-cache
  - prefix-caching
  - speculative-decoding
  - llm-runtime
  - pytorch
  - pure-pytorch
  - vllm
datasets:
  - prajwalpujari16/llm-serving-traces
models:
  - Qwen/Qwen2.5-0.5B-Instruct
  - Qwen/Qwen2.5-1.5B-Instruct
---

# mini-infer 🚀 Paged KV-Cache & Radix Prefix Studio

Interactive simulation of GPU PagedAttention block pool allocation, copy-on-write radix tree prefix caching, speculative decoding verification, and analytical KV-cache sizing.

* **Interactive Features:** Interactive virtual block table allocator, copy-on-write radix prefix caching simulator, and speculative decoding acceptance verifier.
* **GitHub Repository:** [Prajwal7-2002/mini-infer](https://github.com/Prajwal7-2002/mini-infer)
* **PyPI:** [`pip install mini-infer`](https://pypi.org/project/mini-infer/)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))
