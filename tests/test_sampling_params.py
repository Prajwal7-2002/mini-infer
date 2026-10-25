"""Unit tests for SamplingParams dataclass and vectorized engine sampling."""

import torch

from mini_infer import InferenceEngine, Qwen2Config, Qwen2ForCausalLM, SamplingParams
from mini_infer.scheduler.continuous import Sequence, SequenceStatus


def create_toy_engine() -> InferenceEngine:
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    return InferenceEngine(
        model=model,
        config=cfg,
        num_blocks=16,
        block_size=16,
        max_batch_size=4,
        chunked_prefill_size=16,
        enable_prefix_caching=False,
        device=torch.device("cpu"),
    )


def test_sampling_params_defaults_and_is_greedy():
    """Verify default values and greedy detection heuristic."""
    sp = SamplingParams()
    assert sp.max_tokens == 128
    assert sp.temperature == 0.0
    assert sp.top_p == 1.0
    assert sp.top_k == -1
    assert sp.min_p == 0.0
    assert sp.stop_token_ids == set()
    assert sp.is_greedy() is True

    sp_sampling = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, min_p=0.05)
    assert sp_sampling.is_greedy() is False
    assert sp_sampling.top_k == 50
    assert sp_sampling.min_p == 0.05


def test_sequence_backward_compatible_properties():
    """Verify Sequence properties synchronize transparently with SamplingParams."""
    sp = SamplingParams(max_tokens=64, temperature=0.5, top_p=0.8, stop_token_ids={99})
    seq = Sequence(prompt_token_ids=[1, 2, 3], sampling_params=sp)

    assert seq.sampling_params is sp
    assert seq.max_tokens == 64
    assert seq.temperature == 0.5
    assert seq.top_p == 0.8
    assert seq.stop_token_ids == {99}

    # Test setter mutation
    seq.max_tokens = 32
    seq.temperature = 0.0
    assert sp.max_tokens == 32
    assert sp.temperature == 0.0
    assert seq.sampling_params.is_greedy() is True


def test_sequence_legacy_kwargs_construct_sampling_params():
    """Verify legacy constructor initializes SamplingParams cleanly."""
    seq = Sequence(
        prompt_token_ids=[1, 2],
        max_tokens=10,
        temperature=0.8,
        top_p=0.95,
        stop_token_ids={15, 20},
    )
    assert isinstance(seq.sampling_params, SamplingParams)
    assert seq.sampling_params.max_tokens == 10
    assert seq.sampling_params.temperature == 0.8
    assert seq.sampling_params.top_p == 0.95
    assert seq.sampling_params.stop_token_ids == {15, 20}


def test_engine_add_request_with_sampling_params():
    """Verify InferenceEngine.add_request accepts SamplingParams and generates tokens."""
    engine = create_toy_engine()
    sp = SamplingParams(max_tokens=3, temperature=0.0, stop_token_ids={999})

    seq = engine.add_request(prompt_token_ids=[5, 10, 15], sampling_params=sp)
    assert seq.sampling_params is sp

    steps = 0
    while engine.scheduler.has_unfinished_sequences() and steps < 10:
        engine.step()
        steps += 1

    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_token_ids) == 3


def test_engine_vectorized_greedy_fast_path():
    """Verify multiple concurrent greedy sequences step through vectorized argmax correctly."""
    engine = create_toy_engine()

    sp1 = SamplingParams(max_tokens=4, temperature=0.0)
    sp2 = SamplingParams(max_tokens=4, temperature=0.0)
    sp3 = SamplingParams(max_tokens=4, temperature=0.0)

    seq1 = engine.add_request(prompt_token_ids=[1, 2], sampling_params=sp1, request_id="g1")
    seq2 = engine.add_request(prompt_token_ids=[3, 4], sampling_params=sp2, request_id="g2")
    seq3 = engine.add_request(prompt_token_ids=[5, 6], sampling_params=sp3, request_id="g3")

    steps = 0
    while engine.scheduler.has_unfinished_sequences() and steps < 15:
        engine.step()
        steps += 1

    assert seq1.status == SequenceStatus.FINISHED
    assert seq2.status == SequenceStatus.FINISHED
    assert seq3.status == SequenceStatus.FINISHED
    assert len(seq1.output_token_ids) == 4
    assert len(seq2.output_token_ids) == 4
    assert len(seq3.output_token_ids) == 4


def test_engine_mixed_sampling_batch():
    """Verify a batch with both greedy and non-greedy requests steps properly."""
    engine = create_toy_engine()

    # One greedy, one stochastic
    sp_greedy = SamplingParams(max_tokens=3, temperature=0.0)
    sp_stochastic = SamplingParams(max_tokens=3, temperature=0.7, top_p=0.9, top_k=10)

    seq_g = engine.add_request(prompt_token_ids=[10, 11], sampling_params=sp_greedy, request_id="g")
    seq_s = engine.add_request(
        prompt_token_ids=[20, 21], sampling_params=sp_stochastic, request_id="s"
    )

    steps = 0
    while engine.scheduler.has_unfinished_sequences() and steps < 15:
        engine.step()
        steps += 1

    assert seq_g.status == SequenceStatus.FINISHED
    assert seq_s.status == SequenceStatus.FINISHED
    assert len(seq_g.output_token_ids) == 3
    assert len(seq_s.output_token_ids) == 3
