"""Measure what prefix caching actually buys: warmed-up TTFT with and without a cache hit.

Both engines are warmed up first, so neither number includes CUDA initialisation.
Each prompt shares a long system prefix and has a distinct question. The script also
checks that cached and uncached runs produce the same greedy tokens.

    uv run python scripts/bench_prefix_cache.py --prefix-tokens 900 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoTokenizer

from mini_infer.engine import InferenceEngine
from mini_infer.models.loader import load_model_from_safetensors


def ttft_and_tokens(engine: InferenceEngine, prompt: list[int], rid: str, n: int, sync):
    seq = engine.add_request(prompt, max_tokens=n, request_id=rid)
    sync()
    t0 = time.perf_counter()
    ttft = None
    while engine.scheduler.has_unfinished_sequences():
        if engine.step() and ttft is None:
            sync()
            ttft = (time.perf_counter() - t0) * 1000
    return ttft, list(seq.output_token_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prefix-tokens", type=int, default=900)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument(
        "--dtype",
        choices=["auto", "float32"],
        default="auto",
        help="float32 makes cached vs uncached output bit-comparable",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.dtype == "float32" else None
    model, cfg = load_model_from_safetensors(args.model, device=device.type, dtype=dtype)

    line = "module_{i}.py: defines class Handler{i} with methods load, save, validate.\n"
    system, i = "Repository notes:\n", 0
    while len(tok.encode(system)) < args.prefix_tokens:
        system += line.format(i=i)
        i += 1

    def prompt(q: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        return tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    questions = [f"What does module_{k}.py define?" for k in range(args.repeats + 1)]
    results = {}
    for cached in (False, True):
        engine = InferenceEngine(
            model=model, config=cfg, num_blocks=1024, device=device, enable_prefix_caching=cached
        )
        ttft_and_tokens(engine, prompt("Say hi."), "warmup", 4, sync)  # warm-up + fills cache
        runs = [
            ttft_and_tokens(engine, prompt(q), f"q{k}", args.max_tokens, sync)
            for k, q in enumerate(questions[1:])
        ]
        results[cached] = runs

    off = statistics.median(t for t, _ in results[False])
    on = statistics.median(t for t, _ in results[True])
    same = sum(a[1] == b[1] for a, b in zip(results[False], results[True], strict=True))
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"{args.model} on {name}, prompt ~{len(prompt(questions[0]))} tokens, {args.repeats} runs"
    )
    print(f"  TTFT no prefix cache : {off:7.1f} ms (median)")
    print(f"  TTFT prefix-cache hit: {on:7.1f} ms (median)  -> {off / on:.1f}x faster")
    print(
        f"  identical greedy output: {same}/{args.repeats} runs ({next(model.parameters()).dtype})"
    )


if __name__ == "__main__":
    main()
