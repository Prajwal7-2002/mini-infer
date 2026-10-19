# DEVLOG

## 2026-10-19: Project Genesis & Architecture (M0)

- Initialized Flagship #2 `mini-infer`.
- Mission: A minimal, readable, high-performance continuous-batching inference engine with paged KV-cache, prefix caching, and speculative decoding for Qwen2.5 (0.5B / 1.5B) in pure PyTorch.
- Hardware budget: 100% free; fits on a 6 GB NVIDIA RTX 3050 Laptop GPU (1.5B target ~3 GB VRAM + 0.5B draft ~1 GB VRAM + 2 GB Paged KV Cache).
- Initialized repository with Apache 2.0 license, `.gitignore`, and staged milestone roadmap.
