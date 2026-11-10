"""Pure-PyTorch dynamic KV-cache quantization and sub-byte bit-packing primitives.

Supports INT8 (1 byte/elem), FP8 E4M3 (1 byte/elem), and sub-byte packed INT4 (0.5 bytes/elem)
with zero external C++/CUDA build requirements.
"""

from typing import Literal
import torch

KVCacheDType = Literal["auto", "fp16", "fp32", "int8", "fp8", "int4"]


def get_kv_cache_bytes_per_elem(dtype_str: str) -> float:
    """Return theoretical memory footprint in bytes per KV element."""
    dt = dtype_str.lower()
    if dt in ("int4", "q4", "uint4"):
        return 0.5
    elif dt in ("int8", "q8", "fp8", "float8", "fp8_e4m3"):
        return 1.0
    elif dt in ("fp16", "float16", "bf16", "bfloat16", "auto"):
        return 2.0
    elif dt in ("fp32", "float32"):
        return 4.0
    else:
        return 2.0


def quantize_int8(
    tensor: torch.Tensor, scale_dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating point tensor to symmetric INT8 with per-token scale factors.

    Args:
        tensor: Unquantized floating point tensor.
        scale_dim: Dimension along which to calculate scale (default: last dim / head_dim).

    Returns:
        Tuple of (int8_quantized_tensor, scale_tensor).
    """
    scale = torch.clamp(tensor.abs().amax(dim=scale_dim, keepdim=True) / 127.0, min=1e-7)
    q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(
    q_tensor: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize INT8 tensor back to target floating-point precision."""
    return q_tensor.to(out_dtype) * scales


def quantize_fp8(
    tensor: torch.Tensor, scale_dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating point tensor to FP8 (E4M3) with dynamic scale factors.

    Uses native torch.float8_e4m3fn when available; gracefully falls back to scaled INT8
    on platforms lacking float8 tensor core instructions.
    """
    if hasattr(torch, "float8_e4m3fn"):
        # E4M3 maximum representable finite magnitude is 448.0
        max_val = 448.0
        scale = torch.clamp(tensor.abs().amax(dim=scale_dim, keepdim=True) / max_val, min=1e-7)
        scaled = torch.clamp(tensor / scale, -max_val, max_val)
        q = scaled.to(torch.float8_e4m3fn)
        return q, scale
    else:
        # Fallback for older PyTorch versions
        return quantize_int8(tensor, scale_dim=scale_dim)


def dequantize_fp8(
    q_tensor: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize FP8 tensor back to target floating-point precision."""
    return q_tensor.to(out_dtype) * scales


def quantize_and_pack_int4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating point tensor to symmetric INT4 and pack 2 values per uint8 byte.

    Memory consumption is reduced by 75% compared to FP16 (from 2 bytes to 0.5 bytes).

    Args:
        tensor: Floating point tensor of shape [..., head_dim] where head_dim is even.

    Returns:
        Tuple of (packed_uint8_tensor, scale_tensor).
        packed_uint8_tensor has shape [..., head_dim // 2].
    """
    head_dim = tensor.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(
            f"INT4 sub-byte bit packing requires an even head_dim, got {head_dim}."
        )

    # 1. Symmetric scale factor over head_dim [-8, 7]
    scale = torch.clamp(tensor.abs().amax(dim=-1, keepdim=True) / 7.0, min=1e-7)

    # 2. Quantize to signed 4-bit range [-8, 7]
    q = torch.clamp(torch.round(tensor / scale), -8, 7).to(torch.int8)

    # 3. Map to unsigned 4-bit [0, 15] for bitwise operations
    q_u4 = (q + 8).to(torch.uint8)

    # 4. Vectorized pack along head_dim (low 4 bits = even, high 4 bits = odd)
    even = q_u4[..., 0::2]
    odd = q_u4[..., 1::2]
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)

    return packed, scale


def unpack_and_dequantize_int4(
    packed: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Unpack uint8 sub-byte tensor and dequantize back to target floating point precision.

    Args:
        packed: uint8 tensor of shape [..., head_dim // 2].
        scales: Scale tensor of shape [..., 1].
        out_dtype: Target floating point dtype (e.g. torch.float16 or torch.float32).

    Returns:
        Dequantized tensor of shape [..., head_dim].
    """
    # 1. Extract low 4 bits and high 4 bits, shift back to [-8, 7]
    even = (packed & 0x0F).to(torch.int8) - 8
    odd = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    # 2. Interleave even and odd slices into original head_dim
    out_shape = list(packed.shape)
    out_shape[-1] = out_shape[-1] * 2
    unpacked = torch.empty(out_shape, device=packed.device, dtype=torch.int8)
    unpacked[..., 0::2] = even
    unpacked[..., 1::2] = odd

    # 3. Scale back to target precision
    return unpacked.to(out_dtype) * scales
