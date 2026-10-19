import asyncio
import json
import time
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mini_infer.engine import InferenceEngine


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-0.5b"
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128
    stream: bool = False


app = FastAPI(title="mini-infer", description="Continuous-Batching LLM Inference Server")
engine: InferenceEngine | None = None
tokenizer = None


def format_qwen_chat(messages: list[ChatMessage]) -> str:
    """Format messages using the Qwen2.5 chat template."""
    formatted = ""
    for msg in messages:
        formatted += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "mini-infer"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-0.5b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mini-infer",
            },
            {
                "id": "Qwen/Qwen2.5-0.5B-Instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mini-infer",
            },
        ],
    }


engine_lock: asyncio.Lock | None = None


def get_engine_lock() -> asyncio.Lock:
    global engine_lock
    if engine_lock is None:
        engine_lock = asyncio.Lock()
    return engine_lock


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    prompt_text = format_qwen_chat(request.messages)
    # Tokenize prompt
    if tokenizer is not None:
        prompt_token_ids = tokenizer.encode(prompt_text)
    else:
        # Fallback character/dummy tokens for mock testing
        prompt_token_ids = [ord(c) % 256 for c in prompt_text]

    stop_token_ids = set()
    if tokenizer is not None:
        if tokenizer.eos_token_id is not None:
            stop_token_ids.add(tokenizer.eos_token_id)
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
                stop_token_ids.add(im_end_id)
        except Exception:
            pass

    seq = engine.add_request(
        prompt_token_ids=prompt_token_ids,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_token_ids=stop_token_ids,
    )

    lock = get_engine_lock()

    if not request.stream:
        # Non-streaming: step until sequence finishes
        while seq.status.value != "FINISHED":
            async with lock:
                engine.step()
            await asyncio.sleep(0.001)

        output_tokens = seq.output_token_ids
        if output_tokens and output_tokens[-1] in seq.stop_token_ids:
            output_tokens = output_tokens[:-1]

        if tokenizer is not None:
            output_text = tokenizer.decode(output_tokens)
        else:
            output_text = "".join(chr(t) for t in output_tokens)

        return {
            "id": f"chatcmpl-{seq.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": seq.num_prompt_tokens,
                "completion_tokens": seq.num_output_tokens,
                "total_tokens": seq.total_tokens,
            },
        }

    # Streaming mode with SSE
    async def stream_generator() -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        emitted_tokens = 0

        while seq.status.value != "FINISHED" or emitted_tokens < len(seq.output_token_ids):
            async with lock:
                engine.step()

            while emitted_tokens < len(seq.output_token_ids):
                token_id = seq.output_token_ids[emitted_tokens]
                emitted_tokens += 1
                if token_id in seq.stop_token_ids:
                    break
                token_str = tokenizer.decode([token_id]) if tokenizer else chr(token_id)

                chunk = {
                    "id": f"chatcmpl-{seq.request_id}",
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_str},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            await asyncio.sleep(0.001)

        usage_chunk = {
            "id": f"chatcmpl-{seq.request_id}",
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [],
            "usage": {
                "prompt_tokens": seq.num_prompt_tokens,
                "completion_tokens": seq.num_output_tokens,
                "total_tokens": seq.total_tokens,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
