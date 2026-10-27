"""Unit and integration tests for quantbench-engine sub-byte quantization in mini-infer."""

import pytest
import torch

from mini_infer.engine import InferenceEngine
from mini_infer.models.config import Qwen2Config
from mini_infer.models.loader import quantize_model
from mini_infer.models.qwen2 import Qwen2ForCausalLM
from mini_infer.scheduler.continuous import SequenceStatus


def create_toy_causal_lm() -> tuple[Qwen2ForCausalLM, Qwen2Config]:
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
    return model, cfg


@pytest.mark.parametrize("scheme", ["int4", "int8", "ternary", "sparse24"])
def test_quantize_model_all_schemes(scheme: str):
    model, cfg = create_toy_causal_lm()

    quant_model, stats = quantize_model(model, scheme=scheme, group_size=32)

    assert stats["converted_layers"] == 14  # 7 linear layers per layer * 2 layers
    assert stats["skipped_layers"] >= 1  # lm_head is preserved
    assert stats["compression_ratio"] > 1.0
    assert stats["scheme"] == scheme

    # Verify forward pass produces expected shape without NaNs
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits, _ = quant_model(x)

    assert logits.shape == (1, 8, cfg.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_quantized_engine_single_sequence():
    model, cfg = create_toy_causal_lm()
    quant_model, stats = quantize_model(model, scheme="int4", group_size=32)

    engine = InferenceEngine(
        model=quant_model,
        config=cfg,
        num_blocks=16,
        block_size=16,
        device=torch.device("cpu"),
    )

    seq = engine.add_request(prompt_token_ids=[10, 20, 30], max_tokens=4, request_id="quant-req-1")

    step_count = 0
    max_steps = 15
    while engine.scheduler.has_unfinished_sequences() and step_count < max_steps:
        engine.step()
        step_count += 1

    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_token_ids) == 4


def test_quantized_engine_batched_paged_decoding():
    model, cfg = create_toy_causal_lm()
    quant_model, stats = quantize_model(model, scheme="int4", group_size=32)

    engine = InferenceEngine(
        model=quant_model,
        config=cfg,
        num_blocks=32,
        block_size=16,
        max_batch_size=3,
        device=torch.device("cpu"),
    )

    s1 = engine.add_request([1, 2, 3], max_tokens=3, request_id="batch-1")
    s2 = engine.add_request([4, 5], max_tokens=4, request_id="batch-2")
    s3 = engine.add_request([6, 7, 8, 9], max_tokens=2, request_id="batch-3")

    step_count = 0
    max_steps = 20
    while engine.scheduler.has_unfinished_sequences() and step_count < max_steps:
        engine.step()
        step_count += 1

    assert s1.status == SequenceStatus.FINISHED
    assert s2.status == SequenceStatus.FINISHED
    assert s3.status == SequenceStatus.FINISHED

    assert len(s1.output_token_ids) == 3
    assert len(s2.output_token_ids) == 4
    assert len(s3.output_token_ids) == 2


def test_quantized_model_preserves_lm_head():
    """Verify that the vocabulary projection head remains unquantized for precision."""
    model, cfg = create_toy_causal_lm()
    quant_model, stats = quantize_model(
        model, scheme="int4", group_size=32, exclude_modules=["lm_head"]
    )

    assert isinstance(quant_model.lm_head, torch.nn.Linear)
    assert not hasattr(quant_model.lm_head, "packed_weight")
