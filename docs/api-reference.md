# API & CLI Reference 🛠️

mini-infer provides both an **OpenAI-compatible HTTP API** (built with FastAPI and streaming Server-Sent Events) and an intuitive **CLI toolkit**.

---

## CLI Reference

### 1. `mini-infer info`
Inspect hardware environment, PyTorch configuration, and VRAM memory allocation plans.

```bash
mini-infer info
```

### 2. `mini-infer serve`
Start an OpenAI-compatible HTTP inference service.

```bash
mini-infer serve [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model`, `-m` | `TEXT` | `Qwen/Qwen2.5-0.5B-Instruct` | Target model name or local path |
| `--host`, `-h` | `TEXT` | `127.0.0.1` | Network interface to bind to |
| `--port`, `-p` | `INT` | `8000` | Port to listen on |
| `--num-blocks` | `INT` | `1024` | Number of physical Paged KV blocks |
| `--block-size` | `INT` | `16` | Token capacity per physical KV block |
| `--device` | `TEXT` | `auto` | Compute device (`cuda`, `cpu`, or `auto`) |
| `--kv-cache-dtype` | `TEXT` | `auto` | KV-cache precision: `auto` or `int8` (50% memory saving) |
| `--quant`, `-q` | `TEXT` | `None` | Sub-byte weight quantization: `int4`, `int8`, `ternary`, `sparse24` (via `quantbench-engine`) |
| `--group-size` | `INT` | `64` | Block size for blockwise quantization (default: 64) |

---

### 3. `mini-infer repomap`
Generate an AST-based repository map skeleton for LLM coding prompts.

```bash
mini-infer repomap [PATH] [OPTIONS]
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `PATH` | `TEXT` | `.` | Target directory path to recursively index |

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--max-tokens`, `-t` | `INT` | `1500` | Target token budget for AST skeleton |
| `--output`, `-o` | `TEXT` | `None` | Optional markdown file path to save skeleton |

---

### 4. `mini-infer chat`
Interactive terminal coding chat with automatic codebase context and prefix caching.

```bash
mini-infer chat [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model`, `-m` | `TEXT` | `Qwen/Qwen2.5-0.5B-Instruct` | Model name or local checkpoint path |
| `--context`, `-c` | `TEXT` | `None` | Directory path to index or existing context markdown file |
| `--max-tokens` | `INT` | `256` | Maximum generation tokens per reply |
| `--device` | `TEXT` | `auto` | Compute device (`cuda`, `cpu`, or `auto`) |
| `--kv-cache-dtype` | `TEXT` | `int8` | KV-cache precision: `auto` or `int8` |
| `--quant`, `-q` | `TEXT` | `None` | Sub-byte weight quantization: `int4`, `int8`, `ternary`, `sparse24` (via `quantbench-engine`) |
| `--group-size` | `INT` | `64` | Block size for blockwise quantization (default: 64) |
| `--memory / --no-memory` | `BOOL` | `True` | Enable persistent Agent Memory recall across turns and sessions |

---

### 5. `mini-infer memory`
Manage persistent Agent Memory (facts, preferences, decisions, and episodic context).

```bash
mini-infer memory [COMMAND] [OPTIONS]
```

**Commands:**

* **`add`**: Add a verified cognitive fact.
  ```bash
  mini-infer memory add "RTX 3050 has 6GB VRAM" --category core --importance 1.5
  ```
* **`list`**: View stored memories with decay and access metrics.
  ```bash
  mini-infer memory list [--category <core|project|decision|episodic>] [--limit 25]
  ```
* **`search`**: Query memories using hybrid BM25 + dense subword feature ranking.
  ```bash
  mini-infer memory search "quantization" [--top-k 5] [--threshold 0.10]
  ```
* **`stats`**: Display storage footprint, SQLite path, and category distribution.
  ```bash
  mini-infer memory stats
  ```
* **`consolidate`**: Run cognitive garbage collection, pruning decayed items while preserving core rules.
  ```bash
  mini-infer memory consolidate [--max-items 500]
  ```
* **`clear`**: Purge persistent memories from database.
  ```bash
  mini-infer memory clear [--category <cat>] [--force]
  ```

---

## HTTP API Reference

### `POST /v1/chat/completions`

Generates an assistant response given a conversational prompt. Compatible with all standard OpenAI SDKs and tools (e.g. LangChain, LlamaIndex, servebench).

#### Request Body (JSON)

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are an expert AI software architect."},
    {"role": "user", "content": "What is the primary difference between prefill and decode phases?"}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 256,
  "stream": true
}
```

#### Streaming Response Format (Server-Sent Events)

When `"stream": true`, responses are emitted incrementally as standard SSE chunks:

```text
data: {"id":"chatcmpl-102","object":"chat.completion.chunk","created":1740000000,"model":"Qwen/Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-102","object":"chat.completion.chunk","created":1740000000,"model":"Qwen/Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":" primary"},"finish_reason":null}]}

data: [DONE]
```

---

### `GET /v1/models`
Returns list of models currently loaded in memory.

```bash
curl http://127.0.0.1:8000/v1/models
```

---

### `GET /health`
Returns server readiness, active sequence count, and free KV block pool utilization.

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "healthy",
  "active_requests": 3,
  "total_blocks": 2048,
  "free_blocks": 1780,
  "kv_cache_utilization": "13.1%"
}
```
