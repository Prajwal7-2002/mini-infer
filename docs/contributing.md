# Contributing to mini-infer 🤝

Thank you for your interest in contributing to `mini-infer`! We welcome contributions from systems engineers, AI researchers, and open-source developers.

Our mission is to build a minimal, transparent, and high-performance continuous-batching LLM inference engine that runs entirely in pure PyTorch on everyday consumer hardware.

---

## 🛠️ Development Setup

`mini-infer` uses [Astral's `uv`](https://docs.astral.sh/uv/) for fast, deterministic dependency management.

```bash
# 1. Fork and clone the repository
git clone https://github.com/Prajwal7-2002/mini-infer.git
cd mini-infer

# 2. Sync all dependencies into an isolated virtualenv
uv sync

# 3. Verify hardware diagnostics
uv run mini-infer info
```

---

## 🏛️ Core Architectural Invariants

Before submitting a PR, please ensure your contribution adheres to these fundamental design rules:

1. **Pure PyTorch (Zero C++ Compilers)**:
   * `mini-infer` must run on any developer's machine without requiring MSVC Build Tools, CMake, or GCC.
   * All memory paging, slot mapping, and prefix indexing must remain implementable in standard PyTorch and NumPy.
2. **Deterministic & Testable**:
   * Every new feature must include accompanying unit tests in `tests/` that run in under 3 seconds.
   * Model outputs must match reference implementations within numerical tolerances ($10^{-4}$ for `bfloat16`/`float32`).
3. **Silicon Efficiency**:
   * Keep memory allocations bounded. Dynamic tensor allocations in the inner decode loop (`engine.step()`) are strictly prohibited to prevent memory fragmentation and GC bubbles.

---

## 🧪 Testing & Code Quality

Before submitting your pull request, run the test and lint suites:

```bash
# Run unit tests
uv run pytest

# Check code formatting and lint rules
uv run ruff check src tests scripts

# Auto-format
uv run ruff format src tests scripts

# Validate documentation build
uv run mkdocs build --strict
```

---

## 🎯 High-Value Contribution Areas

We actively welcome contributions in the following domains:

* **New Model Architectures**: Implementing self-contained, native `.safetensors` weight loaders for architectures like Llama-3, Gemma-2, or Mistral.
* **Quantization & Cache Compression**: Adding native FP8 KV-cache formats or int4 activation scales.
* **Apple Silicon & AMD Support**: Validating and tuning performance on Apple MPS and AMD ROCm backends.
* **Optional Accelerated Kernels**: Providing drop-in, optional Triton kernels for PagedAttention while maintaining the pure PyTorch fallback path.
* **Scheduler Policies**: Implementing dynamic priority scheduling, Fair-Share batching, or deadline-aware chunked prefill.

---

## 📬 Pull Request Process

1. **Create a branch**: Use a descriptive name like `feat/llama3-loader` or `fix/prefix-cache-hash`.
2. **Follow Conventional Commits**: e.g., `feat:`, `fix:`, `docs:`, `perf:`, `test:`.
3. **Include Silicon Benchmarks**: If your PR makes a performance or memory claim, please include empirical measurements (e.g., TTFT, memory savings, or tok/s) from your local hardware.
4. **Keep PRs Focused**: Avoid giant refactors across unrelated modules in a single PR.
