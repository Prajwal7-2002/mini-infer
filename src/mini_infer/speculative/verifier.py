
import torch
import torch.nn.functional as F

from mini_infer.models.qwen2 import Qwen2ForCausalLM


def speculative_step(
    target_model: Qwen2ForCausalLM,
    draft_model: Qwen2ForCausalLM,
    prefix_ids: torch.Tensor,
    k_draft: int = 3,
    temperature: float = 0.0,
) -> tuple[list[int], int]:
    """Execute one round of Speculative Decoding (Leviathan et al. 2023).

    1. Draft model proposes k_draft candidate tokens autoregressively.
    2. Target model evaluates all k_draft positions in a SINGLE parallel forward pass.
    3. Rejection sampling accepts/rejects candidates with zero distribution drift.

    Args:
        target_model: Larger, accurate target model (e.g. Qwen2.5-1.5B).
        draft_model: Small, fast draft model (e.g. Qwen2.5-0.5B).
        prefix_ids: Tensor of shape [1, seq_len] with current token sequence.
        k_draft: Number of speculative candidate tokens to propose (typically 2-4).
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        Tuple of (newly_generated_tokens, num_accepted_draft_tokens).
    """
    curr_ids = prefix_ids.clone()

    # ----------------------------------------------------
    # Phase 1: Draft Proposer (Autoregressive on small model)
    # ----------------------------------------------------
    draft_tokens: list[int] = []
    draft_probs: list[torch.Tensor] = []

    for _ in range(k_draft):
        with torch.no_grad():
            draft_logits, _ = draft_model(curr_ids)
            last_logits = draft_logits[:, -1, :]

            if temperature <= 1e-5:
                # Greedy: draft token is argmax
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
                p = F.softmax(last_logits, dim=-1)
            else:
                scaled = last_logits / temperature
                p = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(p, num_samples=1)

            draft_tokens.append(next_tok.item())
            draft_probs.append(p)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)

    # ----------------------------------------------------
    # Phase 2: Target Verifier (ONE parallel forward pass!)
    # ----------------------------------------------------
    # Target model evaluates the entire proposed sequence [prefix + k_draft tokens]
    with torch.no_grad():
        target_logits, _ = target_model(curr_ids)

    # Target logits at positions [seq_len - 1, seq_len, ..., seq_len + k_draft - 1]
    # Corresponding to predictions for draft_tokens[0], draft_tokens[1], ..., and the bonus token
    start_pos = prefix_ids.shape[1] - 1
    eval_logits = target_logits[:, start_pos:, :]

    # ----------------------------------------------------
    # Phase 3: Rejection Sampling Verification
    # ----------------------------------------------------
    accepted_tokens: list[int] = []
    num_accepted = 0

    if temperature <= 1e-5:
        # Greedy Speculative Decoding:
        # Accept draft token if target model's argmax matches draft token exactly
        for i in range(k_draft):
            target_argmax = torch.argmax(eval_logits[:, i, :], dim=-1).item()
            if target_argmax == draft_tokens[i]:
                # Candidate accepted!
                accepted_tokens.append(draft_tokens[i])
                num_accepted += 1
            else:
                # Candidate rejected! Take the target model's correction and stop.
                accepted_tokens.append(target_argmax)
                return accepted_tokens, num_accepted

        # All k_draft tokens were accepted! Add bonus token from the (k_draft)-th position
        bonus_token = torch.argmax(eval_logits[:, k_draft, :], dim=-1).item()
        accepted_tokens.append(bonus_token)
        return accepted_tokens, num_accepted

    else:
        # Stochastic Speculative Decoding with Rejection Sampling:
        for i in range(k_draft):
            target_p = F.softmax(eval_logits[:, i, :] / temperature, dim=-1)
            q = draft_probs[i]
            token_id = draft_tokens[i]

            p_tok = target_p[0, token_id].item()
            q_tok = q[0, token_id].item()

            # Acceptance probability
            accept_prob = min(1.0, p_tok / max(q_tok, 1e-10))
            if torch.rand(1).item() < accept_prob:
                accepted_tokens.append(token_id)
                num_accepted += 1
            else:
                # Rejection: Sample from adjusted distribution max(0, p - q)
                adjusted_p = torch.clamp(target_p - q, min=0.0)
                sum_p = adjusted_p.sum(dim=-1, keepdim=True)
                if sum_p.item() > 0:
                    adjusted_p = adjusted_p / sum_p
                    corrected_token = torch.multinomial(adjusted_p, num_samples=1).item()
                else:
                    corrected_token = torch.multinomial(target_p, num_samples=1).item()

                accepted_tokens.append(corrected_token)
                return accepted_tokens, num_accepted

        # If all k_draft were accepted, sample bonus token from target's last position
        last_target_p = F.softmax(eval_logits[:, k_draft, :] / temperature, dim=-1)
        bonus_token = torch.multinomial(last_target_p, num_samples=1).item()
        accepted_tokens.append(bonus_token)
        return accepted_tokens, num_accepted
