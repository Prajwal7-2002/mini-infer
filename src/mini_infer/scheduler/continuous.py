import uuid
from enum import Enum

from mini_infer.cache.block_pool import BlockPool, BlockTable


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
    ):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_token_ids = stop_token_ids or set()

        self.status = SequenceStatus.WAITING
        self.block_table = BlockTable(block_size=16)
        self.past_key_values = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

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
    """Iteration-level scheduler implementing continuous batching.

    Instead of waiting for all requests in a batch to finish (static batching),
    the scheduler checks for finished requests and admits new waiting requests
    at every single decoding iteration step.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        max_batch_size: int = 8,
        max_num_batched_tokens: int = 2048,
    ):
        self.block_pool = block_pool
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add_sequence(self, sequence: Sequence):
        """Enqueue a new request."""
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def has_unfinished_sequences(self) -> bool:
        """Returns True if there are requests currently running or waiting."""
        return bool(self.waiting or self.running)

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
            # Estimate blocks needed for prefill
            blocks_needed = (candidate.num_prompt_tokens + 15) // 16
            if self.block_pool.num_free < blocks_needed:
                # GPU KV pool has no space right now; stop admitting until running sequences retire
                break

            candidate = self.waiting.pop(0)
            candidate.status = SequenceStatus.RUNNING
            admitted.append(candidate)

        # 3. Newly admitted sequences undergo prefill; existing running sequences undergo decode
        prefill_seqs = admitted
        decode_seqs = list(self.running)

        # Add newly admitted sequences into running pool for future steps
        self.running.extend(admitted)

        return prefill_seqs, decode_seqs
