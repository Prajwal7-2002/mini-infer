# IDE & Client Integration Guide 💻

How to connect `mini-infer` to Cursor, VS Code Continue, Aider, Cline, and custom OpenAI clients.

---

## Overview

`mini-infer` implements standard OpenAI-compatible endpoints:
* `POST /v1/chat/completions` (Server-Sent Events streaming)
* `GET /v1/models`

Any tool that works with OpenAI, vLLM, or Ollama works with `mini-infer`.

---

## 1. VS Code (Continue Extension)

1. Install the **Continue** extension from the VS Code Marketplace.
2. Open `~/.continue/config.json` (or click the gear icon in Continue).
3. Add `mini-infer` as a model provider:

```json
{
  "models": [
    {
      "title": "mini-infer Local",
      "provider": "openai",
      "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

---

## 2. Cursor IDE

1. Open **Cursor Settings** (`Ctrl + Shift + J` or `Cmd + Shift + J`).
2. Navigate to **Models**.
3. Under **OpenAI API Key**, toggle **Override OpenAI Base URL**:
   * **Base URL**: `http://127.0.0.1:8000/v1`
   * **API Key**: `dummy` (or any string)
4. Add model name: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (or `Qwen/Qwen2.5-0.5B-Instruct`).

---

## 3. Aider (Terminal Pair Programming)

[Aider](https://aider.chat/) is an autonomous terminal-based AI pair programmer.

Run Aider pointing to your local `mini-infer` server:

```bash
# Set environment variables
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy

# Launch Aider
aider --model openai/Qwen/Qwen2.5-Coder-1.5B-Instruct
```

On Windows PowerShell:
```powershell
$env:OPENAI_API_BASE = "http://127.0.0.1:8000/v1"
$env:OPENAI_API_KEY = "dummy"
aider --model openai/Qwen/Qwen2.5-Coder-1.5B-Instruct
```

---

## 4. Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful software engineer."},
        {"role": "user", "content": "Write a Python decorator that retries on exception."},
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## 5. Terminal `curl` Streaming

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain Paged Attention in 2 sentences."}
    ],
    "stream": true
  }'
```
