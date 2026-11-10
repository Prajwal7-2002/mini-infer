# Dynamic FP8 & INT4 KV-Cache Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement pure-PyTorch dynamic FP8 and sub-byte packed INT4 KV-cache compression in `mini-infer`, cutting KV-cache memory consumption by 50% to 75% on consumer and cloud GPUs with zero external C++/CUDA toolchains.

**Architecture:** A modular `mini_infer.cache.quantizer` provides vectorized, zero-loop quantization/packing/unpacking routines for FP8 (E4M3) and INT4 (sub-byte packing 2 values per `uint8` byte). `BlockPool` integrates these routines transparently in `write_slot`, `write_sequence_kv`, and `read_sequence_kv`, allocating physical KV cache buffers scaled down to `head_dim // 2` for INT4. `InferenceEngine` and `cli.py` expose `kv_cache_dtype in ("auto", "int8", "fp8", "int4")`.

**Tech Stack:** Python 3.12, PyTorch 2.6 (torch.float8_e4m3fn, torch.int8, torch.uint8 bitwise operations), Typer, Rich, Pytest.

**Spec:** `docs/plans/2026-11-10-dynamic-kv-cache-compression.md`

## Global Constraints

- **Pure PyTorch**: Zero C++ toolchains, zero CMake, zero compilation required at install or runtime.
- **Vectorized Performance**: Zero Python element-wise loops during quantization and bit-packing/unpacking; strictly vectorized tensor slicing (`0::2`, `1::2`) and bitwise ops (`&`, `|`, `>>`, `<<`).
- **Memory Reduction**: 
  - `int8` / `fp8`: 1 byte per element (50% reduction vs FP16).
  - `int4`: 0.5 bytes per element (`uint8` packed tensor of shape `head_dim // 2`) + per-token scale factor (75% reduction vs FP16).
- **Exact Drop-in Compatibility**: Must maintain 100% backward compatibility with existing `BlockPool`, `BlockTable`, `PrefixCache`, and `InferenceEngine` APIs. All 51 existing tests must continue to pass.

## Review Focus

1. **Odd `head_dim` handling**: Standard LLM head dimensions are even (64, 128, 256). Verify assertion/guard for even `head_dim` during INT4 packing.
2. **Underflow & Scale Div-by-Zero**: Ensure `torch.clamp(..., min=1e-7)` avoids division by zero on zero-padded slots.
3. **Prefix Caching Alignment**: Ensure prefix hashing and copy-on-write refcounts behave identically regardless of `kv_cache_dtype`.
4. **End-to-End Logits Divergence**: Verify that output logits generated with INT4/FP8 KV-cache have cosine similarity > 0.98 compared to unquantized FP16 KV-cache.
5. **Memory Footprint Assertion**: Verify physical tensor `element_size() * numel()` is mathematically 4x smaller for INT4 vs FP16.

---

### Task 1: Core KV-Cache Compression Primitives (`mini_infer.cache.quantizer`)

**Files:**
- Create: `src/mini_infer/cache/quantizer.py`
- Test: `tests/test_kv_cache_compression.py`

- [ ] **Step 1: Write failing test for KV quantization primitives**
  Write tests in `tests/test_kv_cache_compression.py`:
  - `test_int8_quantization_roundtrip`: Test symmetric int8 per-token quantization and reconstruction error.
  - `test_fp8_quantization_roundtrip`: Test FP8 (E4M3) quantization with fallback handling.
  - `test_int4_pack_unpack_roundtrip`: Test bit-accurate INT4 symmetric sub-byte packing (`uint8` halved dimension) and unpacking.
  - `test_int4_memory_reduction`: Assert that packed tensor byte size is exactly 50% of INT8 and 25% of FP16.

- [ ] **Step 2: Run test to verify it fails**
  Run `pytest tests/test_kv_cache_compression.py` (should fail with `ModuleNotFoundError`).

- [ ] **Step 3: Implement `mini_infer/cache/quantizer.py`**
  Implement:
  - `quantize_int8(tensor: torch.Tensor, scale_dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]`
  - `dequantize_int8(q_tensor: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor`
  - `quantize_fp8(tensor: torch.Tensor, scale_dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]`
  - `dequantize_fp8(q_tensor: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor`
  - `quantize_and_pack_int4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`
  - `unpack_and_dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor`
  - `get_kv_cache_bytes_per_elem(dtype_str: str) -> float`

- [ ] **Step 4: Run tests and ensure Task 1 passes**
  Run `pytest tests/test_kv_cache_compression.py`.

- [ ] **Step 5: Commit Task 1**
  Commit with message `feat(cache): implement pure-PyTorch FP8 and INT4 sub-byte KV-cache compression primitives`.

---

### Task 2: Integrate FP8 and INT4 into `BlockPool`

**Files:**
- Modify: `src/mini_infer/cache/block_pool.py:22-85` (`__init__`)
- Modify: `src/mini_infer/cache/block_pool.py:140-165` (`write_slot`)
- Modify: `src/mini_infer/cache/block_pool.py:180-210` (`write_sequence_kv`)
- Modify: `src/mini_infer/cache/block_pool.py:220-250` (`read_sequence_kv`)
- Test: `tests/test_kv_cache_compression.py`

- [ ] **Step 1: Write failing tests for BlockPool with `int4` and `fp8`**
  Add tests to `tests/test_kv_cache_compression.py`:
  - `test_block_pool_int4_allocation_and_memory`: Check memory footprint of `key_caches` is cut by 75%.
  - `test_block_pool_int4_write_read_slot`: Single-slot write and read accuracy.
  - `test_block_pool_int4_vectorized_sequence_write_read`: Vectorized multi-token scatter and gather.
  - `test_block_pool_fp8_allocation_and_write_read`: FP8 block pool read/write.

- [ ] **Step 2: Run tests to verify failures**
  Run `pytest tests/test_kv_cache_compression.py` (should fail on unsupported `int4`/`fp8`).

- [ ] **Step 3: Update `BlockPool` in `src/mini_infer/cache/block_pool.py`**
  - In `__init__`: Allow `"int4"` (allocate `(num_blocks, num_kv_heads, block_size, head_dim // 2)` with `torch.uint8`) and `"fp8"`.
  - Add method `get_kv_cache_memory_bytes() -> dict[str, int]`.
  - In `write_slot`: Delegate to quantizer based on `self.kv_cache_dtype`.
  - In `write_sequence_kv`: Perform vectorized scatter and pack.
  - In `read_sequence_kv`: Gather and unpack back to `[1, num_kv_heads, num_tokens, head_dim]` in `self.dtype`.

- [ ] **Step 4: Run tests and ensure Task 2 passes**
  Run `pytest tests/test_kv_cache_compression.py` and `pytest tests/test_cache.py`.

- [ ] **Step 5: Commit Task 2**
  Commit with message `feat(cache): integrate FP8 and sub-byte INT4 into Paged BlockPool with 75% VRAM savings`.

---

### Task 3: End-to-End Inference Engine & Continuous Batching Verification

**Files:**
- Modify: `src/mini_infer/engine.py` (validate options & documentation)
- Test: `tests/test_kv_cache_compression.py`

- [ ] **Step 1: Write integration tests for `InferenceEngine` with `int4` and `fp8`**
  Add tests:
  - `test_engine_single_request_int4`: Full prompt prefill + auto-regressive decode generation with INT4 KV cache.
  - `test_engine_single_request_fp8`: Full prompt prefill + decode with FP8 KV cache.
  - `test_engine_batched_continuous_int4`: 4 concurrent requests competing for INT4 KV blocks.
  - `test_engine_prefix_caching_with_int4`: Radix prefix cache hits with INT4 blocks.
  - `test_engine_logits_cosine_similarity`: Compare token logits between `auto` (FP32/FP16) and `int4` / `fp8`, asserting cosine similarity > 0.98.

- [ ] **Step 2: Run tests to verify failures/passes**
  Run `pytest tests/test_kv_cache_compression.py`.

- [ ] **Step 3: Update `InferenceEngine` if needed**
  Ensure logging/diagnostics report KV compression ratio and active KV dtype.

- [ ] **Step 4: Commit Task 3**
  Commit with message `feat(engine): support end-to-end continuous batching with compressed FP8/INT4 KV-cache`.

---

### Task 4: CLI Integration, Diagnostics & Full Regression Test

**Files:**
- Modify: `src/mini_infer/cli.py`
- Modify: `src/mini_infer/__init__.py`
- Test: All tests in `tests/`

- [ ] **Step 1: Update `mini_infer/cli.py`**
  - Expand `--kv-cache-dtype` choices: `("auto", "int8", "fp8", "int4")`.
  - Update help strings showing exact memory savings (`int8: 50%, fp8: 50%, int4: 75%`).
  - Add memory telemetry output on startup displaying exact physical KV-cache VRAM in MB.

- [ ] **Step 2: Run full regression test suite**
  Run `pytest tests/` (all existing 51 tests + new compression tests must pass 100%).

- [ ] **Step 3: Run formatting/linter check**
  Run `ruff check src tests` or equivalent.

- [ ] **Step 4: Commit Task 4 and push**
  Commit with message `feat(cli): expose INT4 and FP8 KV-cache compression in CLI and server`.

