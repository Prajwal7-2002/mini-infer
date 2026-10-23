import torch

from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.prefix_cache import PrefixCache
from mini_infer.models.config import Qwen2Config
from mini_infer.models.qwen2 import Qwen2ForCausalLM
from mini_infer.models.sampler import sample_next_token
from mini_infer.scheduler.continuous import (
    ContinuousScheduler,
    Sequence,
    SequenceStatus,
)


class InferenceEngine:
    """End-to-End LLM Serving Engine combining Model, Paged KV-Cache, Prefix Cache, and Continuous Batching."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        config: Qwen2Config,
        num_blocks: int = 256,
        block_size: int = 16,
        max_batch_size: int = 8,
        chunked_prefill_size: int = 512,
        enable_prefix_caching: bool = True,
        device: torch.device | None = None,
        kv_cache_dtype: str = "auto",
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.config = config
        self.device = device
        self.block_size = block_size
        self.chunked_prefill_size = chunked_prefill_size
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_dtype = kv_cache_dtype

        # 1. Paged KV memory pool
        self.block_pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
            kv_cache_dtype=kv_cache_dtype,
        )

        # 2. Prefix Cache (Radix / Hash based)
        self.prefix_cache = PrefixCache(block_size=block_size) if enable_prefix_caching else None

        # 3. Continuous Batching Scheduler
        self.scheduler = ContinuousScheduler(
            block_pool=self.block_pool,
            max_batch_size=max_batch_size,
            chunked_prefill_size=chunked_prefill_size,
        )

    def add_request(
        self,
        prompt_token_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_token_ids: set[int] | None = None,
        request_id: str | None = None,
    ) -> Sequence:
        """Submit a new request to the waiting queue."""
        seq = Sequence(
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
            request_id=request_id,
        )
        self.scheduler.add_sequence(seq)
        return seq

    def step(self) -> list[tuple[Sequence, int]]:
        """Execute a single continuous batching iteration step.

        Returns:
            List of (sequence, newly_generated_token_id) for each active request in this step.
        """
        prefill_seqs, decode_seqs = self.scheduler.schedule()
        step_outputs: list[tuple[Sequence, int]] = []

        # 1. Handle Chunked Prefill Sequences
        for seq in prefill_seqs:
            input_tokens = seq.prompt_token_ids

            # Prefix caching check on initial admission
            if seq.prompt_tokens_processed == 0 and self.prefix_cache:
                cached_blocks, num_cached = self.prefix_cache.match_prefix(input_tokens, self.block_pool)
                if num_cached > 0:
                    seq.block_table.physical_block_ids.extend(cached_blocks)
                    seq.block_table.num_tokens = num_cached
                    seq.prompt_tokens_processed = num_cached

            # Determine next slice of tokens to process
            remaining_tokens = input_tokens[seq.prompt_tokens_processed:]
            if not remaining_tokens:
                # Fully cached prompt
                chunk_tokens = [input_tokens[-1]]
                seq.block_table.num_tokens -= 1
            else:
                chunk_tokens = remaining_tokens[: self.chunked_prefill_size]

            # Allocate block slots for chunk tokens
            for _ in chunk_tokens:
                seq.block_table.append_slot(self.block_pool)

            # Model forward pass for this prefill chunk
            input_tensor = torch.tensor([chunk_tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, past_kv = self.model(input_tensor, past_key_values=seq.past_key_values)
            seq.past_key_values = past_kv

            # Synchronize prefilled KV states into physical BlockPool
            if past_kv is not None:
                for layer_idx, (k_layer, v_layer) in enumerate(past_kv):
                    self.block_pool.write_sequence_kv(
                        layer_idx=layer_idx,
                        physical_block_ids=seq.block_table.physical_block_ids,
                        start_token_idx=0,
                        k_seq=k_layer[0],
                        v_seq=v_layer[0],
                    )

            seq.prompt_tokens_processed += len(chunk_tokens)

            # If prefill just completed, sample the first output token
            if seq.is_prefill_complete:
                next_token = sample_next_token(
                    logits[:, -1, :],
                    temperature=seq.temperature,
                    top_p=seq.top_p,
                ).item()

                seq.append_token(next_token)
                step_outputs.append((seq, next_token))

                # Register completed prompt blocks into prefix cache
                if self.prefix_cache:
                    full_blocks_count = len(input_tokens) // self.block_size
                    for b_idx in range(len(seq.block_table.physical_block_ids[:full_blocks_count])):
                        block_id = seq.block_table.physical_block_ids[b_idx]
                        prefix_chunk = tuple(input_tokens[: (b_idx + 1) * self.block_size])
                        self.prefix_cache.insert_prefix_block(prefix_chunk, block_id, self.block_pool)

        # 2. Handle Batched Decode Sequences (Single Unified Forward Pass)
        active_decodes = [s for s in decode_seqs if s.status != SequenceStatus.FINISHED]
        if active_decodes:
            # Memory pressure protection: preempt newest running sequence if free blocks exhausted
            while self.block_pool.num_free < len(active_decodes) and len(active_decodes) > 1:
                preempted = active_decodes.pop()
                self.scheduler.preempt(preempted)

            if active_decodes:
                slot_mapping = []
                batch_tokens = []
                for seq in active_decodes:
                    block_id, slot_offset = seq.block_table.append_slot(self.block_pool)
                    slot_mapping.append((block_id, slot_offset))
                    batch_tokens.append([seq.output_token_ids[-1]])

                input_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)
                block_tables = [s.block_table.physical_block_ids for s in active_decodes]
                context_lens = [s.total_tokens for s in active_decodes]

                # Unified forward pass across all concurrent sequences
                with torch.no_grad():
                    logits, _ = self.model(
                        input_tensor,
                        block_pool=self.block_pool,
                        block_tables=block_tables,
                        context_lens=context_lens,
                        slot_mapping=slot_mapping,
                    )

                # Sample next token for each sequence in the batch
                for idx, seq in enumerate(active_decodes):
                    next_tok = sample_next_token(
                        logits[idx, -1, :],
                        temperature=seq.temperature,
                        top_p=seq.top_p,
                    ).item()
                    seq.append_token(next_tok)
                    step_outputs.append((seq, next_tok))

        return step_outputs
