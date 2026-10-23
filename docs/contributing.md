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

## 📬 Pull Request Rules & Submission Process

To keep the codebase maintainable and lightning-fast, all contributions must follow these strict rules:

### 1. Branch Naming Conventions
Always create a descriptive branch off `main`:
* `feat/short-feature-name` (e.g. `feat/llama3-loader`)
* `fix/short-bug-description` (e.g. `fix/radix-hash-collision`)
* `perf/short-perf-description` (e.g. `perf/fp8-kv-cache`)
* `docs/short-doc-description` (e.g. `docs/api-guide`)
* `test/short-test-description` (e.g. `test/chunked-prefill-coverage`)

### 2. Conventional Commit Standards
Format commit messages with clear conventional commit types:
```text
feat(cache): add dynamic FP8 KV-cache quantization
fix(scheduler): prevent race condition when freeing block 0
perf(engine): optimize slot mapping tensor gather
docs(quickstart): add Cursor IDE configuration guide
test(repomap): add test cases for multi-level inheritance
```

### 3. Empirical Silicon Proof Required
If your pull request makes any latency, memory, or throughput claims:
* You **must** provide before-and-after numbers measured on real hardware.
* State your GPU model, VRAM usage, TTFT, and throughput (`tok/s`).
* PRs claiming "massive speedups" without empirical evidence will not be reviewed.

### 4. Pull Request Template Checklist
When opening a PR on GitHub, our automated PR template will prompt you to verify:
- [x] Pure PyTorch Rule maintained (no C++ compilers / CMake required).
- [x] Zero dynamic allocations inside the inner decode step (`engine.step()`).
- [x] All unit tests pass (`uv run pytest`).
- [x] Linter and formatter pass (`uv run ruff check` & `uv run ruff format`).
- [x] Documentation updated and builds strictly (`uv run mkdocs build --strict`).

### 5. Instant Rejection Rules (What We Cannot Accept)
* **C++ / CUDA Build Dependencies**: Any PR that breaks standard `pip install` on a machine without Microsoft Visual C++ Build Tools or GCC will be closed.
* **Breaking Untested Code**: PRs modifying core memory managers without comprehensive unit test coverage in `tests/`.
* **Megapulls**: PRs containing hundreds of lines of unrelated refactors across multiple subsystems. Keep PRs small and atomic.

---
