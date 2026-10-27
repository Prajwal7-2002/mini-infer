import json
from pathlib import Path
from typing import Any

import torch

from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM

DT_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "I64": torch.int64,
}


def quantize_model(
    model: Qwen2ForCausalLM,
    scheme: str = "int4",
    group_size: int = 64,
    target_modules: list[str] | None = None,
    exclude_modules: list[str] | None = None,
    device: str | torch.device | None = None,
) -> tuple[Qwen2ForCausalLM, dict[str, Any]]:
    """In-place weight quantization using quantbench-engine.

    Replaces target linear projection layers (Q/K/V/O attention projections
    and MLP SwiGLU gate/up/down projections) with sub-byte packed QuantLinear buffers.

    Args:
        model: Target Qwen2ForCausalLM model instance.
        scheme: Quantization scheme ('int4', 'int8', 'ternary', 'sparse24').
        group_size: Block size for blockwise quantization (default: 64).
        target_modules: Optional list of layer names to target.
        exclude_modules: Module names to preserve unquantized (defaults to ['lm_head']).
        device: Optional target device to ensure all buffers and bias parameters reside on.

    Returns:
        Tuple of (quantized_model, compression_statistics).
    """
    try:
        from quantbench.nn import convert_model_to_quant
    except ImportError as exc:
        raise ImportError(
            "quantbench-engine is required for sub-byte quantization. "
            "Install it with `pip install quantbench-engine` or `uv add quantbench-engine`."
        ) from exc

    if exclude_modules is None:
        exclude_modules = ["lm_head"]

    model, stats = convert_model_to_quant(
        model,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        scheme=scheme,
        group_size=group_size,
    )

    if device is not None:
        model = model.to(device)

    return model, stats


def _load_safetensors_into_model(
    model: torch.nn.Module,
    sf_path: Path,
    target_dtype: torch.dtype,
) -> None:
    """Stream weights directly into model parameters in-place to avoid duplicate VRAM allocations."""
    import struct

    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    with open(sf_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        data_base_offset = 8 + header_size

        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            length = end - start
            f.seek(data_base_offset + start)
            raw = f.read(length)

            src_dt = DT_MAP.get(info["dtype"], torch.float16)
            t = torch.frombuffer(bytearray(raw), dtype=src_dt).reshape(info["shape"])

            if name in param_dict:
                param_dict[name].data.copy_(t.to(dtype=target_dtype))
            elif name in buffer_dict:
                buffer_dict[name].data.copy_(t.to(dtype=target_dtype))
            del raw, t


def load_model_from_safetensors(
    model_dir_or_repo: str | Path,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    quant_scheme: str | None = None,
    group_size: int = 64,
) -> tuple[Qwen2ForCausalLM, Qwen2Config]:
    """Load Qwen2.5 weights directly from safetensors with optional sub-byte quantization.

    Args:
        model_dir_or_repo: Local folder path or HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').
        device: Target torch device ('cuda', 'cpu').
        dtype: Target torch dtype (torch.bfloat16, torch.float16, torch.float32).
        quant_scheme: Optional quantization scheme ('int4', 'int8', 'ternary', 'sparse24').
        group_size: Group size for blockwise quantization (default: 64).

    Returns:
        Tuple of (model, config).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

    path = Path(model_dir_or_repo)

    # If it's a HuggingFace Hub repo id and not an existing local directory, download via huggingface_hub
    if not path.exists():
        from huggingface_hub import snapshot_download

        print(f"Downloading model {model_dir_or_repo} from Hugging Face Hub...")
        path = Path(
            snapshot_download(
                repo_id=str(model_dir_or_repo),
                allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
            )
        )

    # 1. Load config.json
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    config = Qwen2Config.from_dict(config_dict)

    # 2. Instantiate model directly with target dtype on target device
    print(
        f"Initializing Qwen2ForCausalLM ({config.num_hidden_layers} layers, hidden_size={config.hidden_size})..."
    )
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(orig_dtype)

    # 3. Locate safetensors files
    safetensor_files = sorted(list(path.glob("*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    print(
        f"Streaming weights from {len(safetensor_files)} safetensors file(s) into {device} ({dtype})..."
    )
    for sf in safetensor_files:
        _load_safetensors_into_model(model, sf, target_dtype=dtype)

    # 4. Handle tied word embeddings
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    # 5. Optional Sub-Byte Quantization via quantbench-engine
    if quant_scheme is not None:
        print(
            f"Applying {quant_scheme.upper()} quantization via quantbench-engine (group_size={group_size})..."
        )
        model, stats = quantize_model(
            model,
            scheme=quant_scheme,
            group_size=group_size,
            device=device,
        )
        print(
            f"Quantization complete: converted {stats['converted_layers']} layers "
            f"({stats['orig_memory_mb']} MB -> {stats['quant_memory_mb']} MB, "
            f"{stats['compression_ratio']}x compression in {stats['duration_sec']}s)"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    return model, config
