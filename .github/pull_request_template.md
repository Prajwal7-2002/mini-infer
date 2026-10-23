## Description
<!-- Briefly describe what this PR does, why it is needed, and link any related issues. -->

Fixes #(issue)

## Type of Change
<!-- Please check all options that apply -->
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (non-breaking change adding functionality)
- [ ] ⚡ Performance optimization (latency, memory, or throughput improvement)
- [ ] 📚 Documentation update (guides, docstrings, README)
- [ ] 🧪 Tests (adding missing tests or refactoring test suites)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)

## Empirical Benchmarks / Hardware Measurements
<!-- If this PR affects performance, memory, or serving latency, provide before/after numbers. -->
* **Hardware**: (e.g., NVIDIA RTX 3050 6GB Laptop GPU / CPU / Apple M-series)
* **Model Tested**: (e.g., Qwen/Qwen2.5-0.5B-Instruct or Qwen2.5-Coder-1.5B)
* **Metrics**:
  * **TTFT**: Before: `____ ms` | After: `____ ms`
  * **Peak VRAM / DRAM**: Before: `____ MB` | After: `____ MB`
  * **Generation Throughput**: Before: `____ tok/s` | After: `____ tok/s`

## Architectural Invariants Checklist
<!-- You MUST verify and check every item below before submitting -->
- [ ] **Pure PyTorch Rule**: Zero C++ compiler, CMake, or Visual Studio build requirements. All logic runs in pure Python/PyTorch.
- [ ] **Bounded Memory**: No dynamic tensor allocations inside the inner decode step (`engine.step()`).
- [ ] **Unit Tests**: Added or updated tests in `tests/` and ran `uv run pytest` (must pass 100%).
- [ ] **Formatting & Linting**: Ran `uv run ruff check src tests scripts` and `uv run ruff format src tests scripts` with zero errors.
- [ ] **Documentation**: Updated `README.md` or `docs/` if modifying CLI flags or user-facing behavior, and verified with `uv run mkdocs build --strict`.
- [ ] **Clean Commits**: Follows conventional commit format (`feat:`, `fix:`, `perf:`, `docs:`, `test:`).
