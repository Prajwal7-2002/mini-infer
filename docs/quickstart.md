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
> mini-infer requires **no C++ compilers or CMake**. It runs entirely using standard PyTorch wheels!

---

## 2. Check Hardware Diagnostics

Run the CLI diagnostic tool to verify CUDA availability, detect your GPU, and display the automated VRAM allocation plan:

```bash
mini-infer info
```

Example output on a laptop GPU:
```text
mini-infer System Diagnostics
PyTorch version: 2.6.0+cu124
CUDA available: True
GPU Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Total VRAM: 6.00 GB

Memory Allocation Plan (RTX 3050 6 GB):
  • Qwen2.5-1.5B (Target Model): ~3.0 GB VRAM
  • Qwen2.5-0.5B (Draft Model):  ~1.0 GB VRAM
  • Paged KV-Cache Memory Pool:  ~1.5 GB VRAM (~2,000 blocks)
  • Safety Buffer:               ~0.5 GB
```

---

## 3. Index Your Project Codebase (`mini-infer repomap`)

Before querying an LLM about your code, extract an AST skeleton (classes, method signatures, exports) into a compact prompt:

```bash
# Generate a ~1000-token repository skeleton
mini-infer repomap ./src --max-tokens 1000 -o repo_context.md
```

See [Coding on Large Codebases](guides/coding-projects.md) for full details.

---

## 4. Launching the Serving Engine

Start an OpenAI-compatible HTTP inference server. Pass `--kv-cache-dtype int8` to cut KV cache memory by 48.4%:

### Standard Serving with INT8 KV-Cache
```bash
mini-infer serve \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --kv-cache-dtype int8 \
  --port 8000
```

### Speculative Decoding (1.5B Target + 0.5B Draft)
```bash
mini-infer serve \
  --target Qwen/Qwen2.5-1.5B-Instruct \
  --draft Qwen/Qwen2.5-0.5B-Instruct \
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
