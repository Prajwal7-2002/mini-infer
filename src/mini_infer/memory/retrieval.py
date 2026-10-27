"""Hybrid BM25 and Dense Feature Retrieval Engine for Agent Memory.

Provides fast, zero-dependency semantic and lexical search across stored memories.
Combines Okapi BM25 keyword matching with L2-normalized subword hashed embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import torch

from mini_infer.memory.schema import MemoryItem


def tokenize(text: str) -> list[str]:
    """Clean and tokenize text into lowercase word tokens and n-grams."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in clean.split() if len(t) > 1]
    return tokens


def compute_dense_embedding(text: str, dim: int = 256) -> torch.Tensor:
    """Generate a deterministic, L2-normalized feature embedding vector from text.

    Uses character trigram and subword feature hashing to capture morphological
    and semantic similarity without requiring external cloud APIs or large model weights.
    """
    tokens = tokenize(text)
    if not tokens:
        return torch.zeros(dim, dtype=torch.float32)

    vec = torch.zeros(dim, dtype=torch.float32)

    # 1. Word-level hashing
    for word in tokens:
        h = hash(word) % dim
        vec[h] += 1.0

        # 2. Character trigrams for subword robustness
        if len(word) >= 3:
            for i in range(len(word) - 2):
                tri = word[i : i + 3]
                h_tri = hash(tri) % dim
                vec[h_tri] += 0.5

    # L2 normalize
    norm = torch.norm(vec, p=2)
    if norm > 1e-6:
        vec = vec / norm

    return vec


class BM25Index:
    """In-memory BM25 Okapi lexical index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.doc_term_freqs: list[Counter] = []
        self.doc_freqs: Counter = Counter()

    def fit(self, docs: Sequence[str]) -> None:
        """Index a collection of text documents."""
        self.corpus_size = len(docs)
        self.doc_lens = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_len = 0
        for doc in docs:
            tokens = tokenize(doc)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_len / max(self.corpus_size, 1)

    def score(self, query: str) -> list[float]:
        """Compute BM25 scores for all indexed documents against query."""
        if self.corpus_size == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return [0.0] * self.corpus_size

        scores = [0.0] * self.corpus_size
        for term in q_tokens:
            if term not in self.doc_freqs:
                continue

            # Inverse Document Frequency (IDF)
            df = self.doc_freqs[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            idf = max(idf, 0.01)

            for idx in range(self.corpus_size):
                tf = self.doc_term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1.0))
                )
                scores[idx] += idf * (numerator / denominator)

        # Normalize scores to [0.0, 1.0] range
        max_score = max(scores) if scores else 0.0
        if max_score > 1e-6:
            scores = [s / max_score for s in scores]

        return scores


class HybridRetriever:
    """Hybrid ranker combining dense subword embeddings, BM25, and recency decay."""

    def __init__(self, alpha: float = 0.55):
        """Args:
        alpha: Weight for dense semantic similarity (1.0 = pure vector, 0.0 = pure BM25).
        """
        self.alpha = alpha

    def rank(
        self,
        query: str,
        items: list[MemoryItem],
        top_k: int = 5,
        min_threshold: float = 0.15,
    ) -> list[tuple[MemoryItem, float]]:
        """Rank memory items against query using hybrid scoring.

        Returns:
            List of (memory_item, combined_score) sorted descending by score.
        """
        if not items:
            return []

        # 1. Lexical BM25 Scores
        docs = [item.content for item in items]
        bm25 = BM25Index()
        bm25.fit(docs)
        bm25_scores = bm25.score(query)

        # 2. Dense Semantic Cosine Scores
        q_vec = compute_dense_embedding(query)
        dense_scores: list[float] = []
        for item in items:
            doc_vec = compute_dense_embedding(item.content)
            cos_sim = float(torch.dot(q_vec, doc_vec).item())
            dense_scores.append(max(cos_sim, 0.0))

        # 3. Fuse scores with importance & time decay
        scored_results: list[tuple[MemoryItem, float]] = []
        for idx, item in enumerate(items):
            s_dense = dense_scores[idx] if idx < len(dense_scores) else 0.0
            s_bm25 = bm25_scores[idx] if idx < len(bm25_scores) else 0.0

            # Base hybrid blend
            base_score = (self.alpha * s_dense) + ((1.0 - self.alpha) * s_bm25)

            # Modulate by item importance and access recency decay
            decay = item.compute_decay()
            final_score = base_score * item.importance * decay

            if final_score >= min_threshold:
                scored_results.append((item, round(final_score, 4)))

        # Sort descending by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
