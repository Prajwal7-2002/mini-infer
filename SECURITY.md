# Security Policy 🔒

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `0.2.x` | :white_check_mark: |
| `0.1.x` | :x: |

---

## Model Weight Safety Guidelines

`mini-infer` is designed from the ground up to protect user workstations and edge devices:

1. **Safetensors Exclusively**:
   - `mini-infer` loads model weights strictly via [Hugging Face `safetensors`](https://github.com/huggingface/safetensors) format.
   - Traditional PyTorch `torch.load()` pickles can execute arbitrary Python bytecode upon deserialization. `mini-infer` **strictly bans unpickling untrusted `.bin` or `.pt` model files**.
2. **Local Execution Invariant**:
   - `mini-infer` operates 100% locally on your machine.
   - It transmits zero prompts, telemetry, embeddings, or memory records over external networks.

---

## Reporting a Vulnerability

If you discover a security vulnerability or potential exploit in `mini-infer`:

* **Please DO NOT file a public issue.**
* Report the issue privately via [GitHub Private Vulnerability Reporting](https://github.com/Prajwal7-2002/mini-infer/security/advisories/new) or email maintainer Prajwal Pujari directly at `prajwalpujari16@gmail.com`.

### Submission Checklist
Please include:
1. Clear description of the vulnerability and attack vector.
2. Step-by-step instructions or minimal proof-of-concept (PoC) code to reproduce.
3. Potential impact on the user's host system or data privacy.

### Response SLA
You will receive an initial response within **48 hours**, followed by regular status updates through patch verification, security advisory publication, and coordinated disclosure.
