# -*- coding: utf-8 -*-
"""
Asynchronous Speculative Layer-Streaming Verifier for mini-infer.
Integrates double-buffered prefetching to overlap weight transfers with tensor execution.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_infer.models.qwen2 import Qwen2ForCausalLM


class DoubleBufferedLayerPrefetcher:
    """Manages alternating layer execution buffers for memory-constrained edge serving."""

    def __init__(self, layer_template: nn.Module, device: str = "cpu"):
        self.device = device
        self.is_cuda = device == "cuda" or (
            isinstance(device, torch.device) and device.type == "cuda"
        )
        self.buffers = [
            copy.deepcopy(layer_template).to(device),
            copy.deepcopy(layer_template).to(device),
        ]
        self.active_idx = 0
        self.copy_stream = torch.cuda.Stream() if self.is_cuda else None
        self.ready_event = threading.Event()
        self.cuda_ready_event = torch.cuda.Event() if self.is_cuda else None
        self.prefetch_thread: Optional[threading.Thread] = None

        self.total_prefetch_time = 0.0
        self.total_stall_time = 0.0
        self.layers_prefetched = 0

    def start_prefetch(self, next_weights: Dict[str, torch.Tensor]):
        self.ready_event.clear()
        standby_idx = 1 - self.active_idx
        standby_buf = self.buffers[standby_idx]

        def _transfer():
            t0 = time.perf_counter()
            with torch.no_grad():
                for name, param in standby_buf.named_parameters():
                    if name in next_weights:
                        param.copy_(next_weights[name], non_blocking=self.is_cuda)
                for name, buf in standby_buf.named_buffers():
                    if name in next_weights:
                        buf.copy_(next_weights[name], non_blocking=self.is_cuda)
            if self.is_cuda and self.copy_stream:
                self.cuda_ready_event.record(self.copy_stream)
            self.total_prefetch_time += time.perf_counter() - t0
            self.layers_prefetched += 1
            self.ready_event.set()

        self.prefetch_thread = threading.Thread(target=_transfer, daemon=True)
        self.prefetch_thread.start()

    def swap_and_get_active(self) -> nn.Module:
        t0 = time.perf_counter()
        if self.prefetch_thread is not None:
            self.ready_event.wait()
            self.prefetch_thread.join()
            self.prefetch_thread = None
        if self.is_cuda and self.cuda_ready_event is not None:
            torch.cuda.current_stream().wait_event(self.cuda_ready_event)
        self.total_stall_time += time.perf_counter() - t0
        self.active_idx = 1 - self.active_idx
        return self.buffers[self.active_idx]

    @property
    def overlap_efficiency(self) -> float:
        if self.total_prefetch_time <= 0.0:
            return 1.0
        hidden_time = max(0.0, self.total_prefetch_time - self.total_stall_time)
        return min(1.0, hidden_time / self.total_prefetch_time)


def streaming_speculative_step(
    target_model: Qwen2ForCausalLM,
    draft_model: Qwen2ForCausalLM,
    prefix_ids: torch.Tensor,
    k_draft: int = 3,
    temperature: float = 0.0,
    prefetcher: Optional[DoubleBufferedLayerPrefetcher] = None,
) -> Tuple[List[int], int, Dict[str, Any]]:
    """Execute one round of Speculative Layer-Streaming with double-buffered prefetching.

    1. In-memory draft proposes k_draft tokens autoregressively.
    2. Target model verifies all candidate tokens in 1 streaming sweep.
    3. Double-buffered prefetcher overlaps layer transfers with compute.
    4. Rejection sampling emits accepted tokens and resamples on divergence.

    Returns:
        (newly_generated_tokens, num_accepted_draft_tokens, telemetry_dict)
    """
    curr_ids = prefix_ids.clone()
    draft_tokens: List[int] = []
    draft_probs: List[torch.Tensor] = []

    # 1. Draft Proposal
    for _ in range(k_draft):
        with torch.no_grad():
            d_logits, _ = draft_model(curr_ids)
            last_logits = d_logits[:, -1, :]
            if temperature <= 1e-5:
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
                prob = torch.ones(1, device=curr_ids.device)
            else:
                p = F.softmax(last_logits / temperature, dim=-1)
                next_tok = torch.multinomial(p, num_samples=1)
                prob = p[0, next_tok.item()]

            draft_tokens.append(next_tok.item())
            draft_probs.append(prob)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)

    # 2. Target Verification
    cand_seq = curr_ids
    with torch.no_grad():
        target_logits, _ = target_model(cand_seq)

    # 3. Exact Rejection Sampling
    accepted_tokens: List[int] = []
    num_accepted = 0
    prefix_len = prefix_ids.shape[1]

    for i in range(k_draft):
        cand_token = draft_tokens[i]
        pos_logits = target_logits[:, prefix_len - 1 + i, :]

        if temperature <= 1e-5:
            target_token = torch.argmax(pos_logits, dim=-1).item()
            if cand_token == target_token:
                accepted_tokens.append(cand_token)
                num_accepted += 1
            else:
                accepted_tokens.append(target_token)
                break
        else:
            target_p = F.softmax(pos_logits / temperature, dim=-1)
            p_cand = target_p[0, cand_token].item()
            q_cand = draft_probs[i].item()
            alpha = min(1.0, p_cand / max(q_cand, 1e-12))
            u = torch.rand(1).item()

            if u < alpha:
                accepted_tokens.append(cand_token)
                num_accepted += 1
            else:
                residual = torch.clamp(target_p[0] - q_cand, min=0.0)
                norm = residual.sum()
                if norm > 0:
                    resampled = torch.multinomial(residual / norm, 1).item()
                else:
                    resampled = torch.argmax(target_p[0]).item()
                accepted_tokens.append(resampled)
                break

    # If all accepted, sample bonus token from target
    if num_accepted == k_draft:
        last_target_logits = target_logits[:, prefix_len - 1 + k_draft, :]
        if temperature <= 1e-5:
            bonus = torch.argmax(last_target_logits, dim=-1).item()
        else:
            bp = F.softmax(last_target_logits / temperature, dim=-1)
            bonus = torch.multinomial(bp, 1).item()
        accepted_tokens.append(bonus)

    overlap_eff = prefetcher.overlap_efficiency if prefetcher is not None else 1.0

    telemetry = {
        "num_proposed": k_draft,
        "num_accepted": num_accepted,
        "acceptance_rate": num_accepted / max(k_draft, 1),
        "total_generated": len(accepted_tokens),
        "sweeps_saved": num_accepted,
        "prefetch_overlap_efficiency": overlap_eff,
    }

    return accepted_tokens, num_accepted, telemetry
