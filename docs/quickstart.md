# Quickstart ⚡

Get an OpenAI-compatible continuous batching engine running on your local GPU in under 60 seconds.

---

## 1. Installation

### From PyPI (Standard)
```bash
pip install mini-infer
```

### With `uv` (Fastest)
```bash
uv pip install mini-infer
```

### Run Directly with `uvx` (Zero Install)
```bash
uvx mini-infer --help
```

### From Source (Development with uv)
```bash
git clone https://github.com/Prajwal7-2002/mini-infer.git
cd mini-infer
uv sync
```

> [!TIP]
> **Running Commands with `uv`**:
> When developing from source or within a virtual environment, prefix commands with `uv run` (e.g. `uv run mini-infer info`), or activate the virtual environment first (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS).

---

## 2. Check Hardware Diagnostics

Run the CLI diagnostic tool to verify CUDA availability, detect your GPU, and display the automated VRAM allocation plan and quantization footprint matrix powered by [`quantbench-engine`](https://pypi.org/project/quantbench-engine/):

```bash
uv run mini-infer info
```

Example output on an entry-level laptop GPU (RTX 3050 6GB):
```text
mini-infer System Diagnostics
PyTorch version: 2.6.0+cu124
CUDA available: True
GPU Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Total VRAM: 6.00 GB

          Model Footprint by Quantization Scheme (quantbench-engine)           
+-----------------------------------------------------------------------------+
| Scheme        |   Precision   | Qwen2.5-0.5B | Qwen2.5-1.5B | Max KV Blocks |
|---------------+---------------+--------------+--------------+---------------|
| FP16 (Dense)  | 16-bit Float  |      0.98 GB |      3.08 GB | ~1,800 blocks |
| INT8 Absmax   | 8-bit Signed  |      0.51 GB |      1.58 GB | ~3,200 blocks |
| INT4 Packed   | 4-bit Nibble  |      0.28 GB |      0.84 GB | ~4,800 blocks |
| BitNet Ternary|   1.58-bit    |      0.16 GB |      0.48 GB | ~5,600 blocks |
| Ampere 2:4    | 50% Structured|      0.54 GB |      1.65 GB | ~3,100 blocks |
+-----------------------------------------------------------------------------+
```

---

## 3. Index Your Project Codebase (`mini-infer repomap`)

Before querying an LLM about your code, extract an AST skeleton (classes, method signatures, exports) into a compact prompt:

```bash
# Generate a ~1000-token repository skeleton
uv run mini-infer repomap ./src --max-tokens 1000 -o repo_context.md
```

See [Coding on Large Codebases](guides/coding-projects.md) for full details.

---

## 4. Launching the Serving Engine

Start an OpenAI-compatible HTTP inference server. Pass `--quant int4` to compress weights by ~3.7× and `--kv-cache-dtype int8` to double KV cache capacity:

### Option A: Sub-Byte INT4 Quantized Serving (Recommended for 6 GB GPUs)
```bash
uv run mini-infer serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --quant int4 \
  --group-size 64 \
  --kv-cache-dtype int8 \
  --port 8000
```

### Option B: Standard Serving with INT8 KV-Cache
```bash
uv run mini-infer serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --kv-cache-dtype int8 \
  --port 8000
```

---

## 5. Connect Your IDE or Query via API

Once running, send requests via `curl` or the standard OpenAI Python client.

### Via `curl` (Streaming Tokens via SSE)

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a concise AI assistant."},
      {"role": "user", "content": "Explain PagedAttention in two sentences."}
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "stream": true
  }'
```

### Via OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Write a fast Python quicksort."}],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

> [!TIP]
> To connect **Cursor**, **VS Code Continue**, or **Aider**, check the [IDE & Client Setup Guide](guides/ide-setup.md).

---

## 6. Running the Unit Tests

mini-infer includes a comprehensive unit test suite covering KV-cache memory managers, dynamic INT8 quantization, and AST repomap logic:

```bash
uv run pytest
# Output: 22 passed in 1.87s
```
