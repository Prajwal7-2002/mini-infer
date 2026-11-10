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
