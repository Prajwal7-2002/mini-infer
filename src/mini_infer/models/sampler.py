import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
) -> torch.Tensor:
    """Sample the next token id from unnormalized logits.

    Args:
        logits: Tensor of shape [batch_size, vocab_size] (typically logits[:, -1, :]).
        temperature: Sampling temperature. 0.0 means greedy argmax.
        top_p: Nucleus sampling threshold (0.0 to 1.0).
        top_k: Top-k tokens to keep (>0).
        min_p: Minimum probability threshold relative to maximum token probability (0.0 to 1.0).

    Returns:
        Tensor of shape [batch_size, 1] containing sampled token IDs.
    """
    # 1. Greedy argmax
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature scaling
    logits = logits / temperature

    # 3. Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, top_k)
        min_val = val[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.full_like(logits, -float("Inf")), logits)

    # 4. Min-P filtering
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1, keepdim=True).values
        threshold = max_prob * min_p
        logits = torch.where(probs < threshold, torch.full_like(logits, -float("Inf")), logits)

    # 5. Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    # 6. Categorical multinomial sampling
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
