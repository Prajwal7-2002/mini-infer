import torch

from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM
from mini_infer.speculative.streaming_verifier import (
    DoubleBufferedLayerPrefetcher,
    streaming_speculative_step,
)


def test_streaming_speculative_identical():
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    prefix = torch.tensor([[10, 20, 30]])
    k_draft = 3

    layer_template = model.model.layers[0]
    prefetcher = DoubleBufferedLayerPrefetcher(layer_template, device="cpu")

    accepted, num_acc, telemetry = streaming_speculative_step(
        target_model=model,
        draft_model=model,
        prefix_ids=prefix,
        k_draft=k_draft,
        temperature=0.0,
        prefetcher=prefetcher,
    )

    assert num_acc == 3
    assert len(accepted) == 4
    assert telemetry["acceptance_rate"] == 1.0
    assert telemetry["sweeps_saved"] == 3
    assert telemetry["prefetch_overlap_efficiency"] >= 0.0


def test_streaming_speculative_rejection():
    torch.manual_seed(42)
    cfg_target = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    cfg_draft = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )

    target_model = Qwen2ForCausalLM(cfg_target)
    draft_model = Qwen2ForCausalLM(cfg_draft)
    target_model.eval()
    draft_model.eval()

    prefix = torch.tensor([[5, 15, 25]])

    accepted, num_acc, telemetry = streaming_speculative_step(
        target_model=target_model,
        draft_model=draft_model,
        prefix_ids=prefix,
        k_draft=3,
        temperature=0.0,
    )

    assert len(accepted) >= 1
    assert 0 <= num_acc <= 3
    assert "acceptance_rate" in telemetry
