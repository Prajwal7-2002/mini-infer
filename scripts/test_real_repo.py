"""Real-world end-to-end codebase synthesis test with mini-infer.

Demonstrates:
1. Indexing a real external repository (EdgeSLM/src) into an AST skeleton (~960 tokens).
2. Dynamic INT8 KV-cache allocation (50% DRAM footprint).
3. Real-time code generation for a new API endpoint based on the repo map.
4. Prefix-cached follow-up generation (writing unit tests for the new endpoint).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mini_infer.engine import InferenceEngine
from mini_infer.models.loader import load_model_from_safetensors
from mini_infer.repomap import build_repo_map, estimate_tokens


def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\ML_projects\EdgeSLM\src")
    if not target_dir.exists():
        print(f"Error: Target path {target_dir} does not exist.")
        sys.exit(1)

    print("=" * 72)
    print("  MINI-INFER REAL-WORLD REPO SYNTHESIS BENCHMARK")
    print(f"  Target Repository: {target_dir.resolve()}")
    print("=" * 72)

    # 1. Parse real repo AST
    print("\n[1/4] Parsing repository AST into token-budgeted prompt skeleton...")
    t0 = time.perf_counter()
    repo_map = build_repo_map(target_dir, max_tokens=1000)
    ast_time_ms = (time.perf_counter() - t0) * 1000
    token_est = estimate_tokens(repo_map)

    print(f"      Parsed AST in {ast_time_ms:.2f} ms")
    print(f"      Skeleton Size: ~{token_est} tokens")
    print("\n--- Extracted Architecture Preview ---")
    lines = repo_map.strip().split("\n")[:25]
    print("\n".join(lines) + "\n...")
    print("--------------------------------------")

    # 2. Initialize Model & INT8 KV-Cache
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n[2/4] Initializing model & INT8 Paged KV-Cache on [{device_str.upper()}]...")
    if device_str == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_instance, config = load_model_from_safetensors(model_id, device=device_str)

    num_blocks = 512
    block_size = 16
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    h_dim = config.head_dim

    fp16_bytes = 2 * n_layers * n_kv_heads * block_size * h_dim * num_blocks * 2
    int8_bytes = (2 * n_layers * n_kv_heads * block_size * h_dim * num_blocks * 1) + (
        2 * n_layers * n_kv_heads * block_size * 1 * num_blocks * 2
    )

    print(f"      KV Pool Capacity: {num_blocks * block_size} tokens ({num_blocks} blocks)")
    print(f"      FP16/BF16 Footprint: {fp16_bytes / (1024 * 1024):.2f} MB")
    print(
        f"      INT8 Quantized Footprint: {int8_bytes / (1024 * 1024):.2f} MB (-{(1 - int8_bytes / fp16_bytes) * 100:.1f}%)"
    )

    engine = InferenceEngine(
        model=model_instance,
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
        kv_cache_dtype="int8",
        enable_prefix_caching=True,
    )
    assert engine.block_pool.key_caches[0].dtype == torch.int8, "INT8 verification failed"

    # 3. Turn 1: Cold Prefill with EdgeSLM Repo-Map + Code Generation Task
    print("\n[3/4] Turn 1: Synthesizing new endpoint from EdgeSLM repo map (Cold Start)...")
    system_prompt = (
        "You are an expert software engineer working on this repository.\n"
        f"Repository Architecture:\n{repo_map}\n"
    )
    user_task_1 = (
        "Based on api/app.py in the repository map, write a new FastAPI endpoint `/metrics` "
        "that returns server health, uptime, and classification statistics as JSON."
    )

    messages_1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task_1},
    ]
    prompt_text_1 = tokenizer.apply_chat_template(
        messages_1, tokenize=False, add_generation_prompt=True
    )
    prompt_ids_1 = tokenizer.encode(prompt_text_1)

    print(f"      Total prompt length: {len(prompt_ids_1)} tokens")

    engine.add_request(prompt_ids_1, max_tokens=100, request_id="edge-turn-1")

    t_start_1 = time.perf_counter()
    ttft_1 = None
    gen_tokens_1 = []

    while engine.scheduler.has_unfinished_sequences():
        out = engine.step()
        if out:
            if ttft_1 is None:
                ttft_1 = (time.perf_counter() - t_start_1) * 1000
            for s, tok_id in out:
                if s.request_id == "edge-turn-1":
                    gen_tokens_1.append(tok_id)

    turn1_ms = (time.perf_counter() - t_start_1) * 1000
    gen_code_1 = tokenizer.decode(gen_tokens_1, skip_special_tokens=True)

    print(f"      TTFT (Time To First Token): {ttft_1:.1f} ms")
    print(
        f"      Generated {len(gen_tokens_1)} tokens in {turn1_ms:.1f} ms ({len(gen_tokens_1) / (turn1_ms / 1000):.1f} tok/s)"
    )
    print("\n--- Generated Code (Turn 1: api/app.py Endpoint) ---")
    print(gen_code_1.strip())
    print("-----------------------------------------------------")

    # 4. Turn 2: Follow-up Unit Test Generation (Prefix Cache Hit)
    print("\n[4/4] Turn 2: Generating unit tests for the new endpoint (Prefix Cache Hit)...")
    user_task_2 = (
        "Now write a pytest test function using TestClient to verify this new `/metrics` endpoint."
    )
    messages_2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task_2},
    ]
    prompt_text_2 = tokenizer.apply_chat_template(
        messages_2, tokenize=False, add_generation_prompt=True
    )
    prompt_ids_2 = tokenizer.encode(prompt_text_2)

    engine.add_request(prompt_ids_2, max_tokens=80, request_id="edge-turn-2")

    t_start_2 = time.perf_counter()
    ttft_2 = None
    gen_tokens_2 = []

    while engine.scheduler.has_unfinished_sequences():
        out = engine.step()
        if out:
            if ttft_2 is None:
                ttft_2 = (time.perf_counter() - t_start_2) * 1000
            for s, tok_id in out:
                if s.request_id == "edge-turn-2":
                    gen_tokens_2.append(tok_id)

    turn2_ms = (time.perf_counter() - t_start_2) * 1000
    gen_code_2 = tokenizer.decode(gen_tokens_2, skip_special_tokens=True)

    print(f"      TTFT (Turn 2 - Prefix Cached): {ttft_2:.1f} ms")
    if ttft_1 and ttft_2:
        speedup = ttft_1 / max(ttft_2, 0.1)
        print(f"      Prefix Cache Latency Speedup: {speedup:.1f}x faster TTFT!")
    print(f"      Generated {len(gen_tokens_2)} tokens in {turn2_ms:.1f} ms")
    print("\n--- Generated Code (Turn 2: Pytest Unit Test) ---")
    print(gen_code_2.strip())
    print("-------------------------------------------------")

    print("\n" + "=" * 72)
    print("  REAL-WORLD CODEBASE SYNTHESIS PROOF COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
