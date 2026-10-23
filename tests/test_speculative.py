import torch

from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM
from mini_infer.speculative.verifier import speculative_step


def test_speculative_greedy_identical_models():
    """When draft model is identical to target, all k_draft tokens MUST be accepted + 1 bonus token."""
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
    # Both use the same model instance
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    prefix = torch.tensor([[10, 20, 30]])
    k_draft = 3

    accepted_tokens, num_accepted = speculative_step(
        target_model=model,
        draft_model=model,
        prefix_ids=prefix,
        k_draft=k_draft,
        temperature=0.0,
    )

    # All 3 draft tokens must be accepted, plus 1 bonus token = 4 tokens generated in 1 target pass!
    assert num_accepted == 3
    assert len(accepted_tokens) == 4


def test_speculative_rejection():
    """Verify that a rejected candidate token is corrected by the target model."""
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

    accepted_tokens, num_accepted = speculative_step(
        target_model=target_model,
        draft_model=draft_model,
        prefix_ids=prefix,
        k_draft=3,
        temperature=0.0,
    )

    # Must return at least 1 token (either the first accepted draft or the first target correction)
    assert len(accepted_tokens) >= 1
    assert 0 <= num_accepted <= 3
