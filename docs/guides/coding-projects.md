# Coding on Large Codebases with mini-infer 🚀

How to index, serve, and code on large multi-file repositories using `mini-infer` on a consumer laptop GPU without running out of memory (OOM) or losing architectural context.

---

## The Challenge of Local Codebase LLMs

When developing with local models on consumer hardware (e.g., 4 GB to 8 GB laptop GPUs), developers face three major bottlenecks:

1. **Context Window Saturation**: Pasting whole repositories into LLM prompts quickly blows past context limits and floods prompts with implementation noise.
2. **KV-Cache Memory Bloat**: In FP16/BF16, storing KV activations for thousands of prompt tokens consumes hundreds of megabytes of scarce VRAM.
3. **Repeated Prefill Latency**: Asking multi-turn questions (*"explain line 12"*, *"now write tests"*) forces standard engines to re-prefill the entire codebase from scratch every time, causing 30–60 second latency stalls.

`mini-infer` solves this with a 3-part systems pipeline:
* **AST Repo-Map (`mini-infer repomap`)**: Compresses entire directories into a dense, typed ~1,000-token skeleton.
* **Dynamic INT8 KV-Cache (`--kv-cache-dtype int8`)**: Drops KV DRAM memory by **48.4%** in pure PyTorch.
* **Radix Prefix Caching**: Reuses KV blocks across turns, reducing follow-up Time-To-First-Token (TTFT) by up to **70×**.

---

## 1. Installation

### From PyPI (Standard)
```bash
pip install mini-infer
```

### With `uv` (Ultra-fast)
```bash
uv pip install mini-infer
```

### Run Anywhere with Zero Install (`uvx`)
```bash
uvx mini-infer --help
```

### From Source (Git)
```bash
git clone https://github.com/Prajwal7-2002/mini-infer.git
cd mini-infer
uv sync
```

---

## 2. Index Your Codebase into an AST Skeleton

Navigate to your target project (e.g. Django app, FastAPI backend, or data science repository) and extract an AST skeleton:

```bash
# Print to terminal
mini-infer repomap ./src --max-tokens 1000

# Or save directly to a markdown file for your prompts
mini-infer repomap ./src --max-tokens 1000 -o repo_context.md
```

### What gets extracted:
* Class declarations, inheritance, and docstrings
* Function signatures, typed parameters, and return annotations
* Module-level exports
* Function bodies and test implementations are stripped to save context

#### Example Output:
```markdown
# Repository Map: src

### `api/routes.py`
```python
class Router:
    def __init__(self, prefix: str = ""): ...
    def register(self, path: str, handler: Callable) -> None: ...
```

### `db/session.py`
```python
def get_db_pool() -> AsyncEngine: ...
```
```

---

## 3. Launch Local Serving with INT8 KV-Cache

Start the OpenAI-compatible HTTP server with INT8 quantization enabled:

```bash
# Qwen2.5-Coder-1.5B (Recommended for coding on 4GB-6GB GPUs)
mini-infer serve \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --kv-cache-dtype int8 \
  --port 8000
```

### Verified VRAM Comparison (8,192 token capacity on RTX 3050):
| KV-Cache Precision | DRAM Allocated | Savings |
| :--- | :---: | :---: |
| **FP16 / BF16** | 96.00 MB | Baseline |
| **INT8 Quantized** | **49.50 MB** | **-48.4%** |

---

## 4. Multi-Turn Coding with Prefix Caching

Once the server is running, you can connect your favorite IDE extension (Cursor, VS Code Continue, Aider) or run `curl`.

### Turn 1: Cold Start (Codebase Ingestion)
You ask: *"Where is authentication handled and how can I add a JWT refresh endpoint?"*
* **Prompt**: Includes `repo_context.md` (~1,000 tokens) + your question.
* **Engine**: Computes prefill for the full prompt and stores the KV blocks in the prefix cache table.
* **TTFT**: ~40s (on laptop GPU).

### Turn 2: Follow-up (Prefix Cache Hit)
You ask: *"Now write the pytest unit test for the refresh logic."*
* **Engine**: Detects that the 1,000-token repository skeleton is already resident in GPU DRAM.
* **TTFT**: **~0.5s – 1.0s (38× – 70× faster!)**
* The model immediately begins generating tokens without re-processing the codebase.

---

## 5. Python SDK Integration

If you are building an automated coding agent, CI reviewer, or custom CLI tool:

```python
from mini_infer import InferenceEngine
from mini_infer.repomap import build_repo_map

# 1. Compress target directory into a 1,000-token skeleton
repo_skeleton = build_repo_map("./my_project/src", max_tokens=1000)

# 2. Boot engine with INT8 KV cache
engine = InferenceEngine(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    kv_cache_dtype="int8",
)

# 3. Stream coding generation
prompt = f"""You are a senior software engineer. Here is the repository skeleton:
{repo_skeleton}

Task: Implement a new caching layer in `cache.py` matching our architecture.
"""

for token in engine.generate_stream(prompt):
    print(token, end="", flush=True)
```
