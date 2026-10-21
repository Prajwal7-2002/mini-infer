# Speculative Decoding ⚡

Autoregressive language model generation is notoriously **memory-bandwidth bound**. mini-infer implements **Speculative Decoding** to generate multiple tokens per forward pass without altering the target model's output distribution.

---

## The Bottleneck: Low Arithmetic Intensity

During the decode phase, generating each individual token requires:
1. Streaming the *entire model parameter weight tensor* (e.g. 3.0 GB for a 1.5B FP16 model) from GPU High-Bandwidth Memory (VRAM) into fast on-chip SRAM cache.
2. Performing matrix-vector multiplication with a batch size of $1$.

The **arithmetic intensity** is approximately:

$$\text{Arithmetic Intensity} \approx \frac{2 \times \text{Parameters}}{\text{Parameter Bytes}} \approx 1.0 \text{ FLOP / Byte}$$

On an RTX 3050 (128-bit memory bus, 192 GB/s bandwidth), loading a 1.5B model takes $\approx 15.6 \text{ ms}$ per token. The GPU compute cores sit largely idle waiting for weights to arrive over the memory bus.

However, scoring **$K$ candidate tokens simultaneously** takes almost the exact same memory-fetch time ($\approx 16 \text{ ms}$), because the weight tensor only needs to be read from VRAM once!

---

## The Speculative Paradigm: Draft & Target Models

mini-infer pairs a tiny **Draft Model** with a high-capacity **Target Model**:

- **Draft Model ($M_{\text{draft}}$)**: **Qwen2.5-0.5B** (~1.0 GB VRAM, ultra-fast generation).
- **Target Model ($M_{\text{target}}$)**: **Qwen2.5-1.5B** (~3.0 GB VRAM, superior reasoning and quality).

```
Step 1: Draft model generates K=4 candidate tokens quickly:
        [x1, x2, x3, x4]

Step 2: Target model evaluates all K candidates in ONE parallel forward pass:
        P_target(x1), P_target(x2), P_target(x3), P_target(x4)

Step 3: Verification (Modified Rejection Sampling):
        x1: Accepted  ✔
        x2: Accepted  ✔
        x3: Rejected  ✖ ──► Sample unbiased token from target residual distribution
        x4: Discarded

Result: 3 tokens produced in the time of a single target forward pass!
```

---

## Mathematical Verification: Exact Distribution Preservation

Speculative decoding is **mathematically lossless**. The output distribution is guaranteed to be identical to sampling directly from the target model.

For each candidate token $x_i \sim P_{\text{draft}}$, accept $x_i$ with probability:

$$\alpha(x_i) = \min\left(1, \frac{P_{\text{target}}(x_i)}{P_{\text{draft}}(x_i)}\right)$$

If token $x_k$ is rejected, generation halts at $x_k$, and a replacement token is sampled from the adjusted normalized residual distribution:

$$P_{\text{corrected}}(x) = \frac{\max\left(0, P_{\text{target}}(x) - P_{\text{draft}}(x)\right)}{\sum_{x'} \max\left(0, P_{\text{target}}(x') - P_{\text{draft}}(x')\right)}$$

---

## Speedup Analysis & Acceptance Rate

The expected number of accepted tokens per step with draft length $K$ and average token acceptance rate $\alpha$ is:

$$\mathbb{E}[\text{Tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

For typical coding and conversational tasks with Qwen2.5-0.5B draft and Qwen2.5-1.5B target:
- **Average Acceptance Rate ($\alpha$)**: **$70\% - 85\%$**
- **Draft Length ($K$)**: $4$
- **Net Speedup**: **$1.8\text{x} - 2.4\text{x}$ lower latency** compared to standard auto-regressive generation.
