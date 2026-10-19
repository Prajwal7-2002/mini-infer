from dataclasses import dataclass


@dataclass
class Qwen2Config:
    """Configuration class for Qwen2.5 model architectures."""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True

    @classmethod
    def qwen2_5_0_5b(cls) -> "Qwen2Config":
        """Default preset for Qwen2.5-0.5B-Instruct."""
        return cls(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            max_position_embeddings=32768,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen2_5_1_5b(cls) -> "Qwen2Config":
        """Default preset for Qwen2.5-1.5B-Instruct."""
        return cls(
            vocab_size=151936,
            hidden_size=1536,
            intermediate_size=8960,
            num_hidden_layers=28,
            num_attention_heads=12,
            num_key_value_heads=2,
            head_dim=128,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            max_position_embeddings=32768,
            tie_word_embeddings=False,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Qwen2Config":
        """Construct config from HuggingFace config.json dictionary."""
        return cls(
            vocab_size=data.get("vocab_size", 151936),
            hidden_size=data.get("hidden_size", 896),
            intermediate_size=data.get("intermediate_size", 4864),
            num_hidden_layers=data.get("num_hidden_layers", 24),
            num_attention_heads=data.get("num_attention_heads", 14),
            num_key_value_heads=data.get("num_key_value_heads", 2),
            head_dim=data.get("head_dim", data.get("hidden_size", 896) // data.get("num_attention_heads", 14)),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=data.get("rope_theta", 1000000.0),
            max_position_embeddings=data.get("max_position_embeddings", 32768),
            tie_word_embeddings=data.get("tie_word_embeddings", True),
        )
