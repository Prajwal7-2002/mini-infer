import torch

from mini_infer.engine import InferenceEngine
from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM
from mini_infer.scheduler.continuous import SequenceStatus


def create_toy_engine(
    max_batch_size: int = 4,
    chunked_prefill_size: int = 16,
    kv_cache_dtype: str = "auto",
) -> InferenceEngine:
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    return InferenceEngine(
        model=model,
        config=cfg,
        num_blocks=32,
        block_size=16,
        max_batch_size=max_batch_size,
        chunked_prefill_size=chunked_prefill_size,
        enable_prefix_caching=True,
        device=torch.device("cpu"),
        kv_cache_dtype=kv_cache_dtype,
    )


def test_engine_batched_paged_decoding():
    engine = create_toy_engine(max_batch_size=3)

    seq1 = engine.add_request(prompt_token_ids=[10, 20, 30], max_tokens=3, request_id="req-1")
    seq2 = engine.add_request(prompt_token_ids=[40, 50], max_tokens=4, request_id="req-2")
    seq3 = engine.add_request(prompt_token_ids=[60, 70, 80, 90], max_tokens=2, request_id="req-3")

    initial_free = engine.block_pool.num_free
    assert initial_free == 32

    # Step engine until completion
    step_count = 0
    max_steps = 20
    total_tokens_generated = 0
    while engine.scheduler.has_unfinished_sequences() and step_count < max_steps:
        outputs = engine.step()
        total_tokens_generated += len(outputs)
        step_count += 1

    assert total_tokens_generated == 3 + 4 + 2

    # Verify all 3 finished
    assert seq1.status == SequenceStatus.FINISHED
    assert seq2.status == SequenceStatus.FINISHED
    assert seq3.status == SequenceStatus.FINISHED

    assert len(seq1.output_token_ids) == 3
    assert len(seq2.output_token_ids) == 4
    assert len(seq3.output_token_ids) == 2

    # Step one more time to clean up
    engine.step()

    # All blocks should be freed back to pool
    assert engine.block_pool.num_free == initial_free


def test_engine_chunked_prefill():
    # Set chunked_prefill_size to 8
    engine = create_toy_engine(max_batch_size=2, chunked_prefill_size=8)

    # 20 prompt tokens -> needs ceil(20/8) = 3 prefill steps
    prompt = list(range(1, 21))
    seq = engine.add_request(prompt_token_ids=prompt, max_tokens=3, request_id="chunked-req")

    # Step 1: processes tokens 0..7 (no output token yet)
    out1 = engine.step()
    assert len(out1) == 0
    assert not seq.is_prefill_complete

    # Step 2: processes tokens 8..15 (no output token yet)
    out2 = engine.step()
    assert len(out2) == 0
    assert not seq.is_prefill_complete

    # Step 3: processes tokens 16..19 (prefill finishes, first token generated!)
    out3 = engine.step()
    assert len(out3) == 1
    assert seq.is_prefill_complete
    assert len(seq.output_token_ids) == 1

    # Step 4: decode step 2
    out4 = engine.step()
    assert len(out4) == 1
    assert len(seq.output_token_ids) == 2

    # Step 5: decode step 3 (finishes max_tokens=3)
    out5 = engine.step()
    assert len(out5) == 1
    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_token_ids) == 3


def test_engine_int8_kv_cache():
    engine = create_toy_engine(max_batch_size=2, kv_cache_dtype="int8")
    assert engine.block_pool.kv_cache_dtype == "int8"
    assert engine.block_pool.key_caches[0].dtype == torch.int8

    seq = engine.add_request(prompt_token_ids=[1, 2, 3, 4], max_tokens=3, request_id="int8-req")

    step_count = 0
    while engine.scheduler.has_unfinished_sequences() and step_count < 10:
        engine.step()
        step_count += 1

    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_token_ids) == 3
