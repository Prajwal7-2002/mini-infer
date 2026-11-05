"""Prefix caching must change speed only, never results.

After a prompt is prefilled through the engine (possibly reusing cached prefix blocks),
the KV stored in the paged pool for that prompt must equal the KV of a plain
full-sequence forward pass. Layer-0 keys catch wrong RoPE positions; deeper layers catch
a suffix that did not attend to the cached prefix. Covers a partial-block suffix, a
fully cached prompt (length a multiple of the block size), chunked prefill, INT8 KV, and
re-running an earlier prompt to detect corruption of shared cached blocks.
"""

import pytest
import torch

from test_engine import create_toy_engine

PREFIX = [(7 * i + 3) % 120 + 1 for i in range(40)]  # 2.5 blocks of 16 tokens
PROMPTS = {
    "a": PREFIX + [5, 9, 11],
    "b": PREFIX + [60, 61, 62, 63, 64],
    "full": PREFIX[:32],  # exactly two blocks: fully cached on its second run
}
ORDER = ["a", "b", "full", "full", "a"]


def _prefill_and_read_kv(engine, prompt, rid):
    seq = engine.add_request(prompt, max_tokens=4, request_id=rid)
    while not seq.is_prefill_complete:
        engine.step()
    kv = [
        engine.block_pool.read_sequence_kv(layer, seq.block_table.physical_block_ids, len(prompt))
        for layer in range(engine.config.num_hidden_layers)
    ]
    while engine.scheduler.has_unfinished_sequences():
        engine.step()
    return kv


def _reference_kv(engine, prompt):
    with torch.no_grad():
        _, past = engine.model(torch.tensor([prompt]))
    return past


@pytest.mark.parametrize("kv_cache_dtype,atol", [("auto", 1e-5), ("int8", 5e-2)])
@pytest.mark.parametrize("chunk", [16, 512])
def test_prefix_cached_kv_matches_full_forward(kv_cache_dtype, atol, chunk):
    engine = create_toy_engine(chunked_prefill_size=chunk, kv_cache_dtype=kv_cache_dtype)
    for i, name in enumerate(ORDER):
        prompt = PROMPTS[name]
        got = _prefill_and_read_kv(engine, prompt, f"req-{i}")
        ref = _reference_kv(engine, prompt)
        for layer, ((k, v), (k_ref, v_ref)) in enumerate(zip(got, ref, strict=True)):
            torch.testing.assert_close(k, k_ref, atol=atol, rtol=0, msg=f"run {i} {name} K{layer}")
            torch.testing.assert_close(v, v_ref, atol=atol, rtol=0, msg=f"run {i} {name} V{layer}")


def test_prefix_cache_is_actually_hit():
    engine = create_toy_engine()
    _prefill_and_read_kv(engine, PROMPTS["a"], "warm")
    shared, n_cached = engine.prefix_cache.match_prefix(PROMPTS["b"], engine.block_pool)
    engine.block_pool.free_blocks(shared)  # undo the probe's retain
    assert n_cached == 32
    seq = engine.add_request(PROMPTS["b"], max_tokens=1, request_id="hit")
    engine.step()
    assert seq.block_table.physical_block_ids[:2] == shared
