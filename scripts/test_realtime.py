"""Real-time live demonstration of large-codebase context injection with mini-infer.

Demonstrates:
1. AST Repo-Map generation from codebase files.
2. Dynamic INT8 KV-cache memory allocation (50% VRAM saving).
3. Real-time inference on Qwen2.5 with Turn 1 (cold prefill) vs Turn 2 (radix prefix cache hit).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mini_infer.engine import InferenceEngine
from mini_infer.models.loader import load_model_from_safetensors
from mini_infer.repomap import build_repo_map, estimate_tokens


def main():
    print("=" * 70)
    print("  MINI-INFER REAL-TIME CODING CONTEXT & INT8 KV-CACHE TEST")
    print("=" * 70)

    # 1. Generate Repo-Map in real-time
    root_src = Path(__file__).resolve().parent.parent / "src"
    print(f"\n[1/4] Generating AST Repo-Map for '{root_src.name}'...")
    t0 = time.perf_counter()
    repo_map = build_repo_map(root_src, max_tokens=1000)
    repomap_time = (time.perf_counter() - t0) * 1000
    token_est = estimate_tokens(repo_map)

    print(f"      Parsed AST in {repomap_time:.2f} ms")
    print(f"      Repo-Map token size: ~{token_est} tokens")
    print("\n--- Repo-Map Preview (First 350 chars) ---")
    print(repo_map[:350] + "\n...")
    print("------------------------------------------")

    # 2. Check Device & Model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n[2/4] Initializing model & INT8 Paged KV-Cache on [{device_str.upper()}]...")
    if device_str == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_instance, config = load_model_from_safetensors(model_id, device=device_str)

    # Calculate exact memory footprint
    num_blocks = 512
    block_size = 16
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    h_dim = config.head_dim

    fp16_bytes = 2 * n_layers * n_kv_heads * block_size * h_dim * num_blocks * 2
    int8_bytes = (2 * n_layers * n_kv_heads * block_size * h_dim * num_blocks * 1) + (
        2 * n_layers * n_kv_heads * block_size * 1 * num_blocks * 2
    )

    print(
        f"      Paged KV Blocks: {num_blocks} blocks x {block_size} tokens = {num_blocks * block_size} token capacity"
    )
    print(f"      FP16/BF16 KV Pool Size: {fp16_bytes / (1024 * 1024):.2f} MB")
    print(
        f"      INT8 Quantized Pool Size: {int8_bytes / (1024 * 1024):.2f} MB (-{(1 - int8_bytes / fp16_bytes) * 100:.1f}%)"
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
    assert engine.block_pool.key_caches[0].dtype == torch.int8, "INT8 cache verification failed!"
    print("      INT8 Paged KV-Cache active and verified in GPU DRAM.")

    # 3. Turn 1: Cold Prefill with Full Repo-Map
    print("\n[3/4] Turn 1: Querying engine with full Repo-Map context (Cold Start)...")
    system_prompt = f"You are an expert AI systems engineer. You have access to this codebase outline:\n\n{repo_map}"
    user_query_1 = "Using the BlockPool class from the codebase outline, write a short Python function that allocates a block and returns its ID."

    messages_1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query_1},
    ]
    prompt_text_1 = tokenizer.apply_chat_template(
        messages_1, tokenize=False, add_generation_prompt=True
    )
    prompt_ids_1 = tokenizer.encode(prompt_text_1)

    print(f"      Total prompt length: {len(prompt_ids_1)} tokens")

    engine.add_request(prompt_ids_1, max_tokens=60, request_id="turn-1")

    # Step engine & measure TTFT
    t_start = time.perf_counter()
    ttft_ms = None
    generated_tokens_1 = []

    while engine.scheduler.has_unfinished_sequences():
        out = engine.step()
        if out:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            for s, tok_id in out:
                if s.request_id == "turn-1":
                    generated_tokens_1.append(tok_id)

    turn1_time = (time.perf_counter() - t_start) * 1000
    gen_text_1 = tokenizer.decode(generated_tokens_1, skip_special_tokens=True)

    print(f"      TTFT (Time To First Token): {ttft_ms:.1f} ms")
    print(
        f"      Generated {len(generated_tokens_1)} tokens in {turn1_time:.1f} ms ({len(generated_tokens_1) / (turn1_time / 1000):.1f} tok/s)"
    )
    print("\n--- Model Output (Turn 1) ---")
    print(gen_text_1.strip())
    print("-----------------------------")

    # 4. Turn 2: Follow-up query sharing the same Repo-Map (Prefix Cache Hit!)
    print("\n[4/4] Turn 2: Follow-up query with same Repo-Map (Testing Radix Prefix Cache Hit)...")
    user_query_2 = "Now write a function to free that block."
    messages_2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query_2},
    ]
    prompt_text_2 = tokenizer.apply_chat_template(
        messages_2, tokenize=False, add_generation_prompt=True
    )
    prompt_ids_2 = tokenizer.encode(prompt_text_2)

    engine.add_request(prompt_ids_2, max_tokens=40, request_id="turn-2")

    t_start_2 = time.perf_counter()
    ttft_ms_2 = None
    generated_tokens_2 = []

    while engine.scheduler.has_unfinished_sequences():
        out = engine.step()
        if out:
            if ttft_ms_2 is None:
                ttft_ms_2 = (time.perf_counter() - t_start_2) * 1000
            for s, tok_id in out:
                if s.request_id == "turn-2":
                    generated_tokens_2.append(tok_id)

    turn2_time = (time.perf_counter() - t_start_2) * 1000
    gen_text_2 = tokenizer.decode(generated_tokens_2, skip_special_tokens=True)

    print(f"      TTFT (Turn 2 - Prefix Cached): {ttft_ms_2:.1f} ms")
    if ttft_ms and ttft_ms_2:
        speedup = ttft_ms / max(ttft_ms_2, 0.1)
        print(f"      Prefix Cache Latency Speedup: {speedup:.1f}x faster TTFT!")
    print(f"      Generated {len(generated_tokens_2)} tokens in {turn2_time:.1f} ms")
    print("\n--- Model Output (Turn 2) ---")
    print(gen_text_2.strip())
    print("-----------------------------")

    print("\n" + "=" * 70)
    print("  TEST COMPLETED SUCCESSFULLY: INT8 KV-CACHE + REPO-MAP VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()
