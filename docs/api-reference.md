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
| `--model` | `TEXT` | `Qwen/Qwen2.5-0.5B-Instruct` | Target model name or local path |
| `--draft` | `TEXT` | `None` | Optional draft model name for speculative decoding |
| `--host` | `TEXT` | `127.0.0.1` | Network interface to bind to |
| `--port` | `INT` | `8000` | Port to listen on |
| `--block-size` | `INT` | `16` | Token capacity per physical KV block |
| `--max-num-seqs` | `INT` | `16` | Max concurrent sequences in active batch |
| `--device` | `TEXT` | `cuda` (or `cpu`) | Compute device |

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
