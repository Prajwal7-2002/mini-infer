"""Tests for pure-PyTorch dynamic FP8 and INT4 sub-byte KV-cache compression."""

import pytest
import torch

from mini_infer.cache.quantizer import (
    dequantize_fp8,
    dequantize_int8,
    get_kv_cache_bytes_per_elem,
    quantize_and_pack_int4,
    quantize_fp8,
    quantize_int8,
    unpack_and_dequantize_int4,
)


def test_get_kv_cache_bytes_per_elem():
    assert get_kv_cache_bytes_per_elem("auto") == 2.0
    assert get_kv_cache_bytes_per_elem("fp16") == 2.0
    assert get_kv_cache_bytes_per_elem("int8") == 1.0
    assert get_kv_cache_bytes_per_elem("fp8") == 1.0
    assert get_kv_cache_bytes_per_elem("int4") == 0.5


def test_int8_quantization_roundtrip():
    torch.manual_seed(42)
    # [tokens, heads, head_dim]
    x = torch.randn(16, 4, 64, dtype=torch.float32)

    q, scale = quantize_int8(x)
    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert scale.shape == (16, 4, 1)

    x_recon = dequantize_int8(q, scale, out_dtype=torch.float32)
    assert x_recon.shape == x.shape

    # Check reconstruction SQNR (Signal-to-Quantization-Noise Ratio)
    noise = x - x_recon
    signal_power = (x**2).mean()
    noise_power = (noise**2).mean()
    sqnr_db = 10 * torch.log10(signal_power / noise_power)
    # 8-bit quantization typically achieves > 35 dB SQNR
    assert sqnr_db.item() > 35.0


def test_fp8_quantization_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(16, 4, 64, dtype=torch.float32)

    q, scale = quantize_fp8(x)
    assert q.shape == x.shape
    assert scale.shape == (16, 4, 1)

    x_recon = dequantize_fp8(q, scale, out_dtype=torch.float32)
    assert x_recon.shape == x.shape

    noise = x - x_recon
    sqnr_db = 10 * torch.log10((x**2).mean() / (noise**2).mean())
    # FP8 typically achieves > 30 dB SQNR
    assert sqnr_db.item() > 28.0


def test_int4_pack_unpack_bit_accurate():
    torch.manual_seed(42)
    # Tensor with even head_dim
    x = torch.randn(8, 2, 64, dtype=torch.float32)

    packed, scale = quantize_and_pack_int4(x)

    # Dimension along head_dim must be exactly halved
    assert packed.dtype == torch.uint8
    assert packed.shape == (8, 2, 32)
    assert scale.shape == (8, 2, 1)

    # Check memory: 8 * 2 * 32 * 1 byte = 512 bytes vs original 8 * 2 * 64 * 4 bytes = 4096 bytes
    assert packed.element_size() * packed.numel() == (x.numel() // 2)

    x_recon = unpack_and_dequantize_int4(packed, scale, out_dtype=torch.float32)
    assert x_recon.shape == x.shape

    # Check correlation
    cos_sim = torch.cosine_similarity(x.flatten(), x_recon.flatten(), dim=0)
    assert cos_sim.item() > 0.98

    # Noise error is strictly bounded
    max_err = (x - x_recon).abs().max()
    assert max_err.item() < 0.5


def test_int4_odd_head_dim_raises():
    x_odd = torch.randn(4, 2, 63, dtype=torch.float32)
    with pytest.raises(ValueError, match="even head_dim"):
        quantize_and_pack_int4(x_odd)


def test_block_pool_int4_allocation_and_memory():
    from mini_infer.cache.block_pool import BlockPool

    # 16 blocks, 16 tokens/block, 2 layers, 4 kv heads, 64 head_dim
    pool_fp16 = BlockPool(
        num_blocks=16,
        block_size=16,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float16,
        kv_cache_dtype="auto",
    )
    pool_int4 = BlockPool(
        num_blocks=16,
        block_size=16,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float16,
        kv_cache_dtype="int4",
    )

    mem_fp16 = pool_fp16.get_kv_cache_memory_bytes()
    mem_int4 = pool_int4.get_kv_cache_memory_bytes()

    # Key cache in INT4 should store uint8 with head_dim // 2 = 32
    assert pool_int4.key_caches[0].dtype == torch.uint8
    assert pool_int4.key_caches[0].shape == (16, 4, 16, 32)

    # Key + Value raw storage bytes must be exactly 25% of FP16 (4x reduction!)
    raw_fp16 = mem_fp16["key_bytes"] + mem_fp16["value_bytes"]
    raw_int4 = mem_int4["key_bytes"] + mem_int4["value_bytes"]
    assert raw_int4 == raw_fp16 // 4


def test_block_pool_int4_write_read_accuracy():
    from mini_infer.cache.block_pool import BlockPool

    pool = BlockPool(
        num_blocks=8,
        block_size=16,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_cache_dtype="int4",
    )

    b0 = pool.allocate_block()
    b1 = pool.allocate_block()

    # Write a sequence of 20 tokens (spreads across b0 and b1)
    torch.manual_seed(101)
    k_seq = torch.randn(2, 20, 32, dtype=torch.float32)
    v_seq = torch.randn(2, 20, 32, dtype=torch.float32)

    pool.write_sequence_kv(
        layer_idx=0,
        physical_block_ids=[b0, b1],
        start_token_idx=0,
        k_seq=k_seq,
        v_seq=v_seq,
    )

    # Read back 20 tokens
    k_read, v_read = pool.read_sequence_kv(
        layer_idx=0,
        physical_block_ids=[b0, b1],
        num_tokens=20,
    )

    assert k_read.shape == (1, 2, 20, 32)
    assert v_read.shape == (1, 2, 20, 32)

    cos_k = torch.cosine_similarity(k_seq.flatten(), k_read.flatten(), dim=0)
    cos_v = torch.cosine_similarity(v_seq.flatten(), v_read.flatten(), dim=0)

    assert cos_k.item() > 0.98
    assert cos_v.item() > 0.98


def test_block_pool_fp8_write_read_accuracy():
    from mini_infer.cache.block_pool import BlockPool

    pool = BlockPool(
        num_blocks=4,
        block_size=16,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_cache_dtype="fp8",
    )

    b0 = pool.allocate_block()
    torch.manual_seed(202)
    k_seq = torch.randn(2, 10, 32, dtype=torch.float32)
    v_seq = torch.randn(2, 10, 32, dtype=torch.float32)

    pool.write_sequence_kv(
        layer_idx=0,
        physical_block_ids=[b0],
        start_token_idx=0,
        k_seq=k_seq,
        v_seq=v_seq,
    )

    k_read, v_read = pool.read_sequence_kv(
        layer_idx=0,
        physical_block_ids=[b0],
        num_tokens=10,
    )

    cos_k = torch.cosine_similarity(k_seq.flatten(), k_read.flatten(), dim=0)
    assert cos_k.item() > 0.98

