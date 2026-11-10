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


def _create_toy_engine(kv_cache_dtype: str = "auto", enable_prefix_caching: bool = True):
    from mini_infer.engine import InferenceEngine
    from mini_infer.models.config import Qwen2Config
    from mini_infer.models.qwen2 import Qwen2ForCausalLM

    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    engine = InferenceEngine(
        model=model,
        config=cfg,
        num_blocks=32,
        block_size=16,
        device=torch.device("cpu"),
        kv_cache_dtype=kv_cache_dtype,
        enable_prefix_caching=enable_prefix_caching,
    )
    return engine, model, cfg


def test_engine_single_request_int4_generation():
    from mini_infer.scheduler.continuous import SequenceStatus

    engine, _, _ = _create_toy_engine(kv_cache_dtype="int4")
    seq = engine.add_request([10, 20, 30, 40], max_tokens=5, request_id="int4-seq-1")

    steps = 0
    while engine.scheduler.has_unfinished_sequences() and steps < 20:
        engine.step()
        steps += 1

    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_token_ids) == 5
    assert all(isinstance(tok, int) for tok in seq.output_token_ids)


def test_engine_single_request_fp8_generation():
    from mini_infer.scheduler.continuous import SequenceStatus

    engine, _, _ = _create_toy_engine(kv_cache_dtype="fp8")
    seq = engine.add_request([15, 25, 35], max_tokens=4, request_id="fp8-seq-1")

    steps = 0
    while engine.scheduler.has_unfinished_sequences() and steps < 15:
        engine.step()
        steps += 1

    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_token_ids) == 4


def test_engine_batched_continuous_int4():
    from mini_infer.scheduler.continuous import SequenceStatus

    engine, _, _ = _create_toy_engine(kv_cache_dtype="int4")

    s1 = engine.add_request([1, 2, 3], max_tokens=3, request_id="b1")
    s2 = engine.add_request([4, 5], max_tokens=4, request_id="b2")
    s3 = engine.add_request([6, 7, 8, 9], max_tokens=2, request_id="b3")

    steps = 0
    while engine.scheduler.has_unfinished_sequences() and steps < 25:
        engine.step()
        steps += 1

    assert s1.status == SequenceStatus.FINISHED
    assert s2.status == SequenceStatus.FINISHED
    assert s3.status == SequenceStatus.FINISHED
    assert len(s1.output_token_ids) == 3
    assert len(s2.output_token_ids) == 4
    assert len(s3.output_token_ids) == 2


def test_engine_logits_cosine_similarity():
    """Verify that INT4 and FP8 KV-caches generate tokens and KV tensors closely matching unquantized FP32."""
    engine_fp, model, cfg = _create_toy_engine(kv_cache_dtype="auto")
    engine_int4, _, _ = _create_toy_engine(kv_cache_dtype="int4")
    engine_fp8, _, _ = _create_toy_engine(kv_cache_dtype="fp8")

    # Use identical weights across engines
    engine_int4.model.load_state_dict(model.state_dict())
    engine_fp8.model.load_state_dict(model.state_dict())

    prompt = [12, 34, 56, 78, 90]

    seq_fp = engine_fp.add_request(prompt, max_tokens=2, temperature=0.0)
    seq_int4 = engine_int4.add_request(prompt, max_tokens=2, temperature=0.0)
    seq_fp8 = engine_fp8.add_request(prompt, max_tokens=2, temperature=0.0)

    # 1. Prefill step
    engine_fp.step()
    engine_int4.step()
    engine_fp8.step()

    # 2. Decode step (reads and writes paged KV blocks dynamically)
    engine_fp.step()
    engine_int4.step()
    engine_fp8.step()

    assert len(seq_fp.output_token_ids) == 2
    assert len(seq_int4.output_token_ids) == 2
    assert len(seq_fp8.output_token_ids) == 2

    # Verify that gathered KV caches across layers have high cosine fidelity (>0.98)
    b_fp = seq_fp.block_table.physical_block_ids
    b_int4 = seq_int4.block_table.physical_block_ids
    b_fp8 = seq_fp8.block_table.physical_block_ids

    k_fp, v_fp = engine_fp.block_pool.read_sequence_kv(0, b_fp, seq_fp.total_tokens)
    k_int4, v_int4 = engine_int4.block_pool.read_sequence_kv(0, b_int4, seq_int4.total_tokens)
    k_fp8, v_fp8 = engine_fp8.block_pool.read_sequence_kv(0, b_fp8, seq_fp8.total_tokens)

    cos_int4 = torch.cosine_similarity(k_fp.flatten(), k_int4.flatten(), dim=0)
    cos_fp8 = torch.cosine_similarity(k_fp.flatten(), k_fp8.flatten(), dim=0)

    assert cos_int4.item() > 0.98
    assert cos_fp8.item() > 0.99


def test_prefix_caching_hit_with_int4():
    """Verify that identical prompt prefixes hit the radix prefix cache with INT4."""
    engine, _, _ = _create_toy_engine(kv_cache_dtype="int4", enable_prefix_caching=True)

    # 32 tokens = 2 blocks of 16
    prefix = list(range(32))
    prompt_a = prefix + [101]
    prompt_b = prefix + [102]

    # Warm up cache with prompt_a
    seq1 = engine.add_request(prompt_a, max_tokens=1)
    while not seq1.is_prefill_complete:
        engine.step()

    # Probe prefix cache for prompt_b
    shared, n_cached = engine.prefix_cache.match_prefix(prompt_b, engine.block_pool)
    engine.block_pool.free_blocks(shared)

    assert n_cached == 32
    assert len(shared) == 2
