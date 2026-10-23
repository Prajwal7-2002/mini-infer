import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.scheduler.continuous import (
    ContinuousScheduler,
    Sequence,
    SequenceStatus,
)


def test_continuous_scheduler_admission_and_retirement():
    pool = BlockPool(
        num_blocks=10,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Max batch size 2
    scheduler = ContinuousScheduler(pool, max_batch_size=2)

    seq_a = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=2, request_id="A")
    seq_b = Sequence(prompt_token_ids=[4, 5, 6], max_tokens=10, request_id="B")
    seq_c = Sequence(prompt_token_ids=[7, 8, 9], max_tokens=5, request_id="C")

    scheduler.add_sequence(seq_a)
    scheduler.add_sequence(seq_b)
    scheduler.add_sequence(seq_c)

    # Iteration 1: Only A and B admitted (max_batch_size=2)
    prefill, decode = scheduler.schedule()
    assert [s.request_id for s in prefill] == ["A", "B"]
    assert decode == []
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting[0].request_id == "C"

    # Simulate token generation step: both generate 1 token
    seq_a.append_token(100)
    seq_b.append_token(200)
    assert seq_a.status == SequenceStatus.RUNNING

    # Iteration 2: A and B decode
    prefill, decode = scheduler.schedule()
    assert prefill == []
    assert [s.request_id for s in decode] == ["A", "B"]

    # Sequence A reaches max_tokens (2)
    seq_a.append_token(101)
    seq_b.append_token(201)
    assert seq_a.status == SequenceStatus.FINISHED
    assert seq_b.status == SequenceStatus.RUNNING

    # Iteration 3: Continuous batching magic:
    # A is retired and freed, C is IMMEDIATELY admitted for prefill, while B continues decoding!
    prefill, decode = scheduler.schedule()
    assert [s.request_id for s in prefill] == ["C"]
    assert [s.request_id for s in decode] == ["B"]


def test_scheduler_chunked_prefill():
    pool = BlockPool(
        num_blocks=10,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Chunk size = 16 tokens
    scheduler = ContinuousScheduler(pool, max_batch_size=2, chunked_prefill_size=16)

    # Prompt with 40 tokens (needs 3 chunks: 16 + 16 + 8)
    long_seq = Sequence(prompt_token_ids=list(range(40)), max_tokens=5, request_id="Long")
    scheduler.add_sequence(long_seq)

    # Step 1: Chunk 1 (16 tokens)
    prefill, decode = scheduler.schedule()
    assert [s.request_id for s in prefill] == ["Long"]
    assert decode == []
    long_seq.prompt_tokens_processed += 16
    assert not long_seq.is_prefill_complete

    # Step 2: Chunk 2 (16 tokens)
    prefill, decode = scheduler.schedule()
    assert [s.request_id for s in prefill] == ["Long"]
    assert decode == []
    long_seq.prompt_tokens_processed += 16
    assert not long_seq.is_prefill_complete

    # Step 3: Chunk 3 (final 8 tokens)
    prefill, decode = scheduler.schedule()
    assert [s.request_id for s in prefill] == ["Long"]
    long_seq.prompt_tokens_processed += 8
    assert long_seq.is_prefill_complete
    long_seq.append_token(999)  # generate first token

    # Step 4: Now in decode mode!
    prefill, decode = scheduler.schedule()
    assert prefill == []
    assert [s.request_id for s in decode] == ["Long"]


def test_scheduler_preemption():
    pool = BlockPool(
        num_blocks=3,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    scheduler = ContinuousScheduler(pool, max_batch_size=2)

    seq_1 = Sequence(prompt_token_ids=[1, 2], max_tokens=10, request_id="S1")
    seq_2 = Sequence(prompt_token_ids=[3, 4], max_tokens=10, request_id="S2")

    scheduler.add_sequence(seq_1)
    scheduler.add_sequence(seq_2)

    prefill, decode = scheduler.schedule()
    assert len(prefill) == 2

    # Allocate blocks for both
    _ = seq_1.block_table.append_slot(pool)
    _ = seq_2.block_table.append_slot(pool)
    assert pool.num_free == 1

    # Preempt S2 under memory pressure
    scheduler.preempt(seq_2)
    assert seq_2.status == SequenceStatus.WAITING
    assert seq_2 not in scheduler.running
    assert scheduler.waiting[0].request_id == "S2"
    assert pool.num_free == 2  # S2's blocks freed back to pool
    assert scheduler.preempted_count == 1
