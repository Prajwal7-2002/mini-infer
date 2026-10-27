# Sub-Byte Weight Quantization & Packaging 📦

In `mini-infer v0.2.6`, model weight representation can be dynamically converted to sub-byte packed integer buffers via direct integration with [`quantbench-engine`](https://pypi.org/project/quantbench-engine/).

---

## The Decode Memory Bandwidth Bottleneck

During autoregressive token generation (decode phase), batch sizes are small ($B=1$ to $B=8$) and sequence length grows by $1$ token per step. In this regime, compute intensity is governed by the **arithmetic intensity** $I$:

$$
I = \frac{\text{Total Floating Point Operations}}{\text{Total DRAM Bytes Transferred}} = \frac{2 \times P \times B}{P \times \text{bytes\_per\_param} + \text{KV\_cache\_bytes}} \approx \frac{2}{\text{bytes\_per\_param}} \quad (\text{FLOPs/byte for } B=1)
$$

On modern consumer GPUs such as the **NVIDIA GeForce RTX 3050 6GB Laptop GPU**:
- **Peak FP32 / FP16 Compute**: $9.1\text{ TFLOPs}$
- **Peak Memory Bandwidth**: $192\text{ GB/s}$ (128-bit GDDR6)
- **Ridge Point**: $\frac{9,100\text{ GFLOPs}}{192\text{ GB/s}} = 47.4\text{ FLOPs/byte}$

Because $I_{\text{FP16}} = \frac{2}{2} = 1.0\text{ FLOP/byte} \ll 47.4$, single-batch decode runs strictly inside the memory-bandwidth-bound region of the hardware roofline. Compressing weights from 16-bit float down to 4-bit integer cuts DRAM traffic by **73.4%**, directly scaling tokens per second on consumer silicon.

---

## Supported Quantization Schemes

`mini-infer` supports 4 drop-in quantization schemes powered by `quantbench-engine`:

| Scheme | Target Bitwidth | Storage Format | Compression Factor | Use Case |
|---|---|---|---|---|
| **`int4`** | 4-bit nibbles | Packed `uint8` byte buffer + blockwise FP32 scales | **~3.7× – 5.0×** | Maximum VRAM reduction for 1.5B/7B models on 6 GB GPUs |
| **`int8`** | 8-bit signed | `torch.int8` + per-channel absmax scales | **~2.0× – 2.9×** | Near-zero perplexity loss serving |
| **`ternary`** | 1.58-bit | 2-bit offset-binary packed buffer + mean absolute $\gamma$ | **~6.0× – 7.5×** | Extreme compression for BitNet architectures |
| **`sparse24`** | 2:4 structured | 50% condensed float values + 2-bit position metadata | **~1.7× – 2.0×** | Hardware-accelerated Ampere sparse Tensor Cores |

---

## Vectorized INT4 Nibble Packing (Zero Python Loops)

Storing 4-bit integers in standard PyTorch tensors without packing causes PyTorch to allocate a full 8-bit or 16-bit container per element, yielding zero memory reduction.

`quantbench-engine` packs two 4-bit signed/unsigned nibbles into each physical `uint8` byte purely via vectorized tensor operations:

```python
# Vectorized uint8 nibble packing (pure PyTorch)
low_nibble = q_int4[..., 0::2] & 0x0F
high_nibble = (q_int4[..., 1::2] & 0x0F) << 4
packed_uint8 = low_nibble | high_nibble
```

During forward execution, `QuantLinear` unpacks and dequantizes blockwise scales on the fly:

```python
# Vectorized unpacking back to float
low = (packed_uint8 & 0x0F).to(torch.int8)
high = ((packed_uint8 >> 4) & 0x0F).to(torch.int8)
unpacked = torch.stack([low, high], dim=-1).reshape(original_shape)
w_hat = (unpacked.to(torch.float32) - zero_points) * scales
```

---

## Precision Architecture: Preserving `lm_head`

Quantizing the final vocabulary projection matrix (`lm_head`, shape `[hidden_size, vocab_size]` = `[1536, 151936]` for Qwen2.5) introduces significant perplexity degradation due to extreme logit dynamic ranges.

`mini-infer`'s `quantize_model()` explicitly excludes `lm_head` from quantization:
- **Quantized**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` across all transformer layers.
- **Unquantized**: `embed_tokens` and `lm_head` remain in full FP16/BF16 precision, ensuring sharp output logit distributions.

---

## Usage Examples

### 1. Serve with INT4 Quantization
```bash
# Using uv run (recommended):
uv run mini-infer serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --quant int4 \
  --group-size 64 \
  --kv-cache-dtype int8 \
  --port 8000
```

### 2. Interactive Terminal Chat with INT4 Weights
```bash
uv run mini-infer chat \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --context ./src \
  --quant int4
```

### 3. Programmatic Usage in Python
```python
from mini_infer.models.loader import load_model_from_safetensors

# Loads Qwen2.5 and automatically converts linear layers to packed INT4
model, config = load_model_from_safetensors(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
    quant_scheme="int4",
    group_size=64,
)
```
