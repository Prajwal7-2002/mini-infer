import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM


def load_model_from_safetensors(
    model_dir_or_repo: str | Path,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Qwen2ForCausalLM, Qwen2Config]:
    """Load Qwen2.5 weights directly from safetensors without transformers.AutoModel overhead.

    Args:
        model_dir_or_repo: Local folder path or HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').
        device: Target torch device ('cuda', 'cpu').
        dtype: Target torch dtype (torch.float16, torch.bfloat16).

    Returns:
        Tuple of (model, config).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    path = Path(model_dir_or_repo)

    # If it's a HuggingFace Hub repo id and not an existing local directory, download via huggingface_hub
    if not path.exists():
        from huggingface_hub import snapshot_download
        print(f"Downloading model {model_dir_or_repo} from Hugging Face Hub...")
        path = Path(snapshot_download(repo_id=str(model_dir_or_repo), allow_patterns=["*.safetensors", "config.json", "tokenizer*"]))

    # 1. Load config.json
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    config = Qwen2Config.from_dict(config_dict)

    # 2. Instantiate model on meta device first or target device
    print(f"Initializing Qwen2ForCausalLM ({config.num_hidden_layers} layers, hidden_size={config.hidden_size})...")
    model = Qwen2ForCausalLM(config)

    # 3. Locate safetensors files
    safetensor_files = sorted(list(path.glob("*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    print(f"Loading weights from {len(safetensor_files)} safetensors file(s) onto {device} ({dtype})...")
    state_dict = {}
    for sf in safetensor_files:
        weights = load_file(str(sf), device="cpu")
        state_dict.update(weights)

    # 4. Handle tied word embeddings if necessary
    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        if "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # 5. Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Ignore lm_head if tie_word_embeddings is True
        if config.tie_word_embeddings and missing == ["lm_head.weight"]:
            pass
        else:
            print(f"Warning: Missing keys: {missing[:5]} (total {len(missing)})")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]} (total {len(unexpected)})")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config
