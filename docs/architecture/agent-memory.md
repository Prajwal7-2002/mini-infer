# Persistent Agent Memory Engine 🧠

`mini-infer` includes an embedded, zero-dependency **Agent Cognitive Memory Engine** (`mini_infer.memory`). It provides long-term recall, hybrid semantic/lexical retrieval, and automatic prompt context injection across terminal chat sessions—preventing context window bloat and eliminating the need for heavy external vector databases.

---

## 1. The Agent Memory Problem

Autonomous AI agents and local inference engines face a fundamental dilemma:

1. **Context Window Limitations**: Modern open-weight models (such as Qwen 2.5 0.5B/1.5B) have practical context budgets where KV-cache VRAM grows linearly with prompt length. Cramming entire chat histories and file trees into every prompt causes out-of-memory (OOM) failures on consumer GPUs.
2. **Session Amnesia**: Standard terminal chats lose all state once the process exits. User rules, hardware specs, and design decisions must be re-explained in every session.
3. **Heavy Vector DB Bloat**: Existing agent memory frameworks (e.g., LangChain, MemGPT, ChromaDB) require multi-gigabyte dependency trees, C++ compilers, external REST servers, or cloud API keys—violating `mini-infer`'s zero-dependency philosophy.

### The Solution: Embedded Zero-Dependency Cognitive Storage
`mini-infer` solves this with an integrated SQLite WAL storage layer combined with a deterministic, subword-hashed dense vector embedder and Okapi BM25 lexical ranker running in pure PyTorch/Python standard library.

```
┌────────────────────────────────────────────────────────────────────────┐
│ mini-infer Interactive Chat / Agent Runner                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ User Query: "What quantization?"
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MemoryEngine Orchestrator                                              │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Hybrid Retriever                                               │   │
│   │                                                                │   │
│   │  Dense Cosine Similarity       Okapi BM25 Lexical Matching     │   │
│   │  (256-dim subword hashing)   + (Term & Inverted Document Freq) │   │
│   │               \                             /                  │   │
│   │                ▼                           ▼                   │   │
│   │           Score Fusion: α · S_dense + (1-α) · S_BM25           │   │
│   │                                 │                              │   │
│   │                                 ▼                              │   │
│   │           Modulate by Importance & Exponential Time Decay      │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                   │
│                                    ▼                                   │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Dynamic Prompt Context Injector                                │   │
│   │ Injects top relevant verified facts into system prompt budget  │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │ ACID Commit / WAL Fetch
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Embedded SQLite Store (.mini_infer/memory.db)                          │
│ Indexed on `category`, `last_accessed`, `id` (Zero network overhead)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Memory Taxonomy

Memories are categorized into a 4-tier cognitive hierarchy:

| Category | Enum | Decay Behavior | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| **Core** | `core` | **None** ($D(t) \equiv 1.0$) | Permanent hardware constraints, developer preferences, safety rules. | `"RTX 3050 has 6GB VRAM; never exceed 2048 KV blocks"` |
| **Project** | `project` | Half-life = 30 days | Codebase architecture, module contracts, CLI signatures. | `"INT4 quantized weights packed in 4-bit uint8 nibbles"` |
| **Decision** | `decision` | Half-life = 21 days | Architectural Decision Records (ADRs) and trade-off rationale. | `"Selected BM25+dense hybrid over pure vector for symbol names"` |
| **Episodic** | `episodic` | Half-life = 7 days | Conversational history, temporary tool outputs, user prompts. | `"User ran servebench against port 8000 on 2026-10-27"` |

---

## 3. Mathematical Foundations

### 3.1. Subword-Hashed Dense Embeddings
To avoid downloading a multi-gigabyte BERT or Sentence-Transformers checkpoint, `mini-infer` computes deterministic 256-dimensional feature embeddings directly in PyTorch:

$$\vec{v}_{\text{raw}} = \sum_{w \in T} \mathbf{e}_{\text{hash}(w) \pmod D} + \frac{1}{2} \sum_{w \in T} \sum_{i=0}^{|w|-3} \mathbf{e}_{\text{hash}(w[i:i+3]) \pmod D}$$

where $D = 256$ is the embedding dimension and $\mathbf{e}_k$ is a unit one-hot basis vector. The resulting vector is $L_2$-normalized:

$$\vec{v} = \frac{\vec{v}_{\text{raw}}}{\|\vec{v}_{\text{raw}}\|_2 + \epsilon}$$

Dense cosine similarity between query $q$ and document $d$ is computed via dot product:

$$S_{\text{dense}}(q, d) = \max\left(0, \vec{v}_q \cdot \vec{v}_d\right)$$

This captures morphological similarity (e.g., `quantization` $\leftrightarrow$ `quant_linear`, `vram` $\leftrightarrow$ `memory`) in **$<0.2\text{ ms}$ on CPU**.

### 3.2. Okapi BM25 Lexical Ranking
Dense embeddings can occasionally miss exact variable names or unique file paths. Okapi BM25 guarantees that exact identifier matches receive top priority:

$$\text{IDF}(t) = \ln\left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1.0 \right)$$

$$S_{\text{BM25}}(q, d) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where default parameters are $k_1 = 1.5$ and $b = 0.75$. BM25 scores are normalized to $[0, 1]$.

### 3.3. Hybrid Score Fusion & Time Decay
The final retrieval score combines dense semantic score, lexical BM25 score, memory importance weight ($W_{\text{imp}} \in [0.1, 2.0]$), and recency decay:

$$S_{\text{final}}(q, d) = \left( \alpha \cdot S_{\text{dense}}(q, d) + (1 - \alpha) \cdot S_{\text{BM25}}(q, d) \right) \times W_{\text{imp}} \times D(t)$$

where default $\alpha = 0.55$.

#### Exponential Decay with Frequency Reinforcement
For non-`CORE` categories, time decay follows an exponential half-life curve boosted by total access hits:

$$D(t) = \exp\left( -\frac{\ln 2}{t_{1/2}} \cdot \Delta t_{\text{days}} \right) \times \left(1.0 + \min(0.5, 0.05 \cdot \text{hits})\right)$$

This mimics human long-term potentiation: frequently accessed memories are retained longer even as time elapses.

---

## 4. Deduplication & Consolidation

When adding memories via `MemoryEngine.add()`, incoming content is checked against all existing memories in the category:

- If cosine similarity $S_{\text{dense}}(\text{new}, \text{existing}) \ge 0.88$, the engine **reinforces the existing memory** (incrementing its access hit count and refreshing its timestamp) rather than creating a duplicate entry.
- `MemoryEngine.consolidate()` runs cognitive garbage collection, automatically pruning decayed memories when storage exceeds target capacity while permanently protecting `CORE` items.

---

## 5. CLI & Chat Usage

### 5.1. Interactive Chat Shortcuts
When running `mini-infer chat`, the memory engine is active by default:

```bash
# Start chat with active Agent Memory
mini-infer chat --model Qwen/Qwen2.5-0.5B-Instruct --memory

# In-chat shortcuts:
User: /remember RTX 3050 has 6GB VRAM and runs INT4 inference
mini-infer: [Memorized] (core): RTX 3050 has 6GB VRAM and runs INT4 inference

User: /memory
mini-infer: Persistent Memories (1 items):
  - [core] RTX 3050 has 6GB VRAM and runs INT4 inference
```

### 5.2. CLI Memory Management
`mini-infer memory` provides complete standalone CRUD and inspection commands:

```bash
# Add a verified fact
mini-infer memory add "Use group_size=64 for INT4 sub-byte quantization" --category project --importance 1.5

# List stored memories
mini-infer memory list

# Search memories using hybrid ranking
mini-infer memory search "quantization settings"

# Display database footprint and telemetry
mini-infer memory stats

# Garbage collect decayed items
mini-infer memory consolidate --max-items 500

# Purge memories
mini-infer memory clear --force
```

### 5.3. Environment Configuration
The storage location defaults to `.mini_infer/memory.db` in the current workspace. To use a custom path or shared global database across projects:

```bash
export MINI_INFER_MEMORY_DB="/path/to/shared_memory.db"
```
