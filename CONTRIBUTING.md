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

## 🎯 Where We Need Help (Areas for Contribution)

We actively welcome contributions in the following areas:

### 1. New Model Architectures (`src/mini_infer/models/loader.py`)
Expand native zero-dependency Safetensors loaders:
- [ ] **Llama-3.2 (1B & 3B)**: Native weight mapping for Meta's GQA edge models with RMSNorm and RoPE.
- [ ] **SmolLM2 (135M / 360M / 1.7B)**: Compact local models for low-VRAM edge devices and CPU execution.
- [ ] **Gemma-2 (2B)**: Support for Gemma-style sliding-window local/global attention patterns.

### 2. Advanced Sampling & Generation (`src/mini_infer/engine.py`)
- [ ] **Min-P Sampling**: Dynamic probability thresholding that scales based on the top token's confidence.
- [ ] **Repetition & Frequency Penalty**: Modifying logits before softmax to prevent generation loops.
- [ ] **JSON / Regex Masking**: Constrain sampling to valid JSON schemas matching Pydantic definitions.

### 3. KV-Cache Innovations (`src/mini_infer/cache.py`)
- [ ] **FP8 KV-Cache (E4M3 / E5M2)**: Native FP8 quantized storage for NVIDIA Ada/Hopper architectures.
- [ ] **StreamingLLM / Attention Sinks**: Keep 4 initial tokens pinned + sliding window of recent tokens for infinite generation without OOM.

### 4. Cross-Platform Acceleration & Tooling
- [ ] **Apple Silicon (MPS)**: Benchmarking and optimizing PagedAttention gather operations on Mac Metal.
- [ ] **Optional Triton Kernels**: Drop-in fused attention kernel with automatic pure-PyTorch fallback.
- [ ] **IDE & Agent Extensions**: Verified configuration recipes and benchmark suites for Aider, Continue, and Cursor.

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

## 📄 Code of Conduct

`mini-infer` is committed to fostering an open, welcoming, and harassment-free community. Please treat all contributors and users with respect and constructive feedback.
