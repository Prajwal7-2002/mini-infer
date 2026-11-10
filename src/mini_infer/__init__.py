"""mini-infer: Minimal continuous-batching inference engine with paged KV-cache."""

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.quantizer import (
    dequantize_fp8,
    dequantize_int8,
    get_kv_cache_bytes_per_elem,
    quantize_and_pack_int4,
    quantize_fp8,
    quantize_int8,
    unpack_and_dequantize_int4,
)
from mini_infer.engine import InferenceEngine
from mini_infer.memory import MemoryCategory, MemoryEngine, MemoryItem
from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM
from mini_infer.models.sampler import SamplingParams
from mini_infer.repomap import build_repo_map

__version__ = "0.3.0"

__all__ = [
    "BlockPool",
    "InferenceEngine",
    "MemoryCategory",
    "MemoryEngine",
    "MemoryItem",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "SamplingParams",
    "build_repo_map",
    "dequantize_fp8",
    "dequantize_int8",
    "get_kv_cache_bytes_per_elem",
    "quantize_and_pack_int4",
    "quantize_fp8",
    "quantize_int8",
    "unpack_and_dequantize_int4",
]
