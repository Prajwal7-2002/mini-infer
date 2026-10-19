import torch

from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import (
    Qwen2ForCausalLM,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from mini_infer.models.sampler import sample_next_token


def test_config():
    cfg_05b = Qwen2Config.qwen2_5_0_5b()
    assert cfg_05b.hidden_size == 896
    assert cfg_05b.num_attention_heads == 14
    assert cfg_05b.num_key_value_heads == 2
    assert cfg_05b.tie_word_embeddings is True

    cfg_15b = Qwen2Config.qwen2_5_1_5b()
    assert cfg_15b.hidden_size == 1536
    assert cfg_15b.num_attention_heads == 12
    assert cfg_15b.num_key_value_heads == 2
    assert cfg_15b.tie_word_embeddings is False


def test_rmsnorm():
    norm = Qwen2RMSNorm(hidden_size=64, eps=1e-6)
    x = torch.randn(2, 10, 64) * 5.0
    out = norm(x)
    assert out.shape == (2, 10, 64)
    # Check that root mean square is close to 1
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rotary_embedding():
    dim = 32
    rope = Qwen2RotaryEmbedding(dim=dim, max_position_embeddings=512, base=10000.0)
    x = torch.randn(1, 4, dim)
    pos_ids = torch.arange(0, 4).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    assert cos.shape == (1, 4, dim)
    assert sin.shape == (1, 4, dim)

    # Test applying rope
    q = torch.randn(1, 2, 4, dim)
    k = torch.randn(1, 2, 4, dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_sampler():
    # Greedy test: highest logit must always win
    logits = torch.tensor([[1.0, 2.0, 10.0, 3.0]])
    token = sample_next_token(logits, temperature=0.0)
    assert token.item() == 2

    # Top-K test
    logits = torch.tensor([[100.0, 99.0, 1.0, 2.0]])
    token = sample_next_token(logits, temperature=1.0, top_k=2)
    assert token.item() in [0, 1]


def test_toy_model_kv_cache_invariant():
    """Verify that cached autoregressive decoding produces the exact same logits as full prefill."""
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=256,
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

    prompt = torch.tensor([[10, 20, 30, 40, 50]])

    # 1. Full forward pass
    with torch.no_grad():
        full_logits, _ = model(prompt)

    # 2. Step-by-step cached generation
    with torch.no_grad():
        # Prefill first 4 tokens
        prefix = prompt[:, :4]
        _, past_kv = model(prefix)

        # Decode token 5 with KV cache
        last_token = prompt[:, 4:5]
        cached_logits, _ = model(last_token, past_key_values=past_kv)

    # The logits for the 5th token must match between full forward and cached step
    assert torch.allclose(full_logits[:, 4:5, :], cached_logits, atol=1e-4), (
        f"Max diff: {(full_logits[:, 4:5, :] - cached_logits).abs().max()}"
    )
