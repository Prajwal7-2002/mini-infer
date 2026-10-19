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
