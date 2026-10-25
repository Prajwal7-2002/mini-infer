import uuid
from enum import Enum

from mini_infer.cache.block_pool import BlockPool, BlockTable
from mini_infer.models.sampler import SamplingParams


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Sequence:
    """Represents a single generation request throughout its lifecycle."""

    def __init__(
        self,
        prompt_token_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_token_ids: set[int] | None = None,
        request_id: str | None = None,
        sampling_params: SamplingParams | None = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []

        if sampling_params is not None:
            self.sampling_params = sampling_params
        else:
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids or set(),
            )

        self.status = SequenceStatus.WAITING
        self.block_table = BlockTable(block_size=16)
        self.past_key_values = None
        self.prompt_tokens_processed: int = 0

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    @max_tokens.setter
    def max_tokens(self, val: int):
        self.sampling_params.max_tokens = val

    @property
    def temperature(self) -> float:
        return self.sampling_params.temperature

    @temperature.setter
    def temperature(self, val: float):
        self.sampling_params.temperature = val

    @property
    def top_p(self) -> float:
        return self.sampling_params.top_p

    @top_p.setter
    def top_p(self, val: float):
        self.sampling_params.top_p = val

    @property
    def stop_token_ids(self) -> set[int]:
        return self.sampling_params.stop_token_ids

    @stop_token_ids.setter
    def stop_token_ids(self, val: set[int]):
        self.sampling_params.stop_token_ids = val

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_output_tokens > 0 or self.prompt_tokens_processed >= self.num_prompt_tokens

    def append_token(self, token_id: int):
        """Append a newly generated token and check if the sequence should terminate."""
        self.output_token_ids.append(token_id)
        if token_id in self.stop_token_ids or self.num_output_tokens >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

    def free(self, block_pool: BlockPool):
        """Release allocated physical blocks back to the memory pool."""
        self.block_table.free(block_pool)
        self.past_key_values = None


class ContinuousScheduler:
    """Iteration-level scheduler implementing continuous batching with chunked prefill and preemption.

    Instead of waiting for all requests in a batch to finish (static batching),
    the scheduler checks for finished requests, admits new waiting requests,
    and interleaves chunked prefills with active decoding streams.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        max_batch_size: int = 8,
        max_num_batched_tokens: int = 2048,
        chunked_prefill_size: int = 512,
    ):
        self.block_pool = block_pool
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.chunked_prefill_size = chunked_prefill_size

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.preempted_count: int = 0

    def add_sequence(self, sequence: Sequence):
        """Enqueue a new request."""
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def has_unfinished_sequences(self) -> bool:
        """Returns True if there are requests currently running or waiting."""
        return bool(self.waiting or self.running)

    def preempt(self, sequence: Sequence) -> None:
        """Preempt a running sequence under memory pressure to avoid OOM crash."""
        if sequence in self.running:
            self.running.remove(sequence)
        sequence.free(self.block_pool)
        sequence.prompt_tokens_processed = 0
        sequence.output_token_ids.clear()
        sequence.status = SequenceStatus.WAITING
        self.waiting.insert(0, sequence)
        self.preempted_count += 1

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """Schedule sequences for the next iteration step.

        Returns:
            Tuple of (prefill_sequences, decode_sequences).
        """
        # 1. Clean up finished running sequences
        still_running = []
        for seq in self.running:
            if seq.status == SequenceStatus.FINISHED:
                seq.free(self.block_pool)
            else:
                still_running.append(seq)
        self.running = still_running

        # 2. Check how many new waiting sequences can be admitted
        admitted: list[Sequence] = []
        while self.waiting and (len(self.running) + len(admitted)) < self.max_batch_size:
            candidate = self.waiting[0]
            # Minimum blocks needed to start chunked prefill
            blocks_needed = min(2, (candidate.num_prompt_tokens + 15) // 16)
            if self.block_pool.num_free < blocks_needed:
                # KV pool is full; stop admitting until older requests complete
                break

            candidate = self.waiting.pop(0)
            candidate.status = SequenceStatus.RUNNING
            admitted.append(candidate)

        self.running.extend(admitted)

        # 3. Separate sequences actively completing prefill vs decode
        prefill_seqs = [s for s in self.running if not s.is_prefill_complete]
        decode_seqs = [s for s in self.running if s.is_prefill_complete]

        return prefill_seqs, decode_seqs
