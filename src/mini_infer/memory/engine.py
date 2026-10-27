"""MemoryEngine: High-level cognitive memory orchestrator for local AI agents.

Manages persistent storage, hybrid semantic retrieval, deduplication, and
automatic prompt context formatting for mini-infer inference sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_infer.memory.retrieval import HybridRetriever, compute_dense_embedding
from mini_infer.memory.schema import MemoryCategory, MemoryItem
from mini_infer.memory.store import SQLiteMemoryStore


class MemoryEngine:
    """Persistent Agent Memory Engine providing long-term recall across sessions."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        retriever_alpha: float = 0.55,
    ):
        self.store = SQLiteMemoryStore(db_path)
        self.retriever = HybridRetriever(alpha=retriever_alpha)

    def add(
        self,
        content: str,
        category: str | MemoryCategory = MemoryCategory.PROJECT,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
        dedup_threshold: float = 0.88,
    ) -> MemoryItem:
        """Add a memory item with duplicate detection.

        If a memory with near-identical semantic content already exists,
        its access count is incremented and timestamp refreshed rather than
        creating a duplicate entry.

        Args:
            content: Raw memory text fact or instruction.
            category: Memory category ('core', 'project', 'decision', 'episodic').
            importance: Priority weight (0.1 to 2.0).
            metadata: Optional dictionary of structured tags.
            dedup_threshold: Cosine similarity threshold for duplicate detection.

        Returns:
            Created or updated MemoryItem.
        """
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")

        cat_enum = (
            category if isinstance(category, MemoryCategory) else MemoryCategory(category.lower())
        )

        # Check existing memories for duplicate
        existing = self.store.list_all(category=cat_enum)
        new_vec = compute_dense_embedding(clean_content)

        import torch

        for item in existing:
            item_vec = compute_dense_embedding(item.content)
            sim = float(torch.dot(new_vec, item_vec).item())
            if sim >= dedup_threshold:
                # Reinforce existing memory
                self.store.update_access(item.id)
                item.access_count += 1
                item.importance = max(item.importance, importance)
                if metadata:
                    item.metadata.update(metadata)
                self.store.insert(item)
                return item

        # Create new item
        item = MemoryItem(
            content=clean_content,
            category=cat_enum,
            importance=max(0.1, min(importance, 2.0)),
            metadata=metadata or {},
        )
        self.store.insert(item)
        return item

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.15,
        category: MemoryCategory | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Search stored memories using hybrid lexical and dense scoring.

        Args:
            query: User prompt or search text.
            top_k: Maximum number of results to return.
            threshold: Minimum score threshold.
            category: Optional category filter.

        Returns:
            List of (MemoryItem, score) pairs sorted by relevance.
        """
        candidates = self.store.list_all(category=category)
        if not candidates:
            return []

        results = self.retriever.rank(query, candidates, top_k=top_k, min_threshold=threshold)

        # Update access telemetry for retrieved items
        for item, _ in results:
            self.store.update_access(item.id)

        return results

    def format_prompt_context(
        self,
        query: str,
        max_tokens: int = 300,
        top_k: int = 4,
    ) -> str:
        """Generate a formatted markdown context block for system prompt injection.

        Retrieves top relevant memories and constructs a clean section:
        ```markdown
        ### Persistent Agent Memory:
        - [Core] Always write typed Python code with pydantic v2.
        - [Project] Target GPU is RTX 3050 (6GB) using INT4 quantization.
        ```
        """
        results = self.search(query, top_k=top_k, threshold=0.12)
        if not results:
            return ""

        lines = ["\n[PERSISTENT AGENT MEMORY - VERIFIED FACTS]"]
        estimated_tokens = 10

        for item, _score in results:
            cat_name = item.category.value.capitalize()
            line = f"- [{cat_name}] {item.content}"
            # Rough token estimate: ~1 token per 4 characters
            item_tokens = len(line) // 4 + 1
            if estimated_tokens + item_tokens > max_tokens:
                break
            lines.append(line)
            estimated_tokens += item_tokens

        lines.append("[END PERSISTENT MEMORY]\n")
        return "\n".join(lines)

    def delete(self, item_id: str) -> bool:
        """Remove a memory item by ID."""
        return self.store.delete(item_id)

    def clear(self, category: str | None = None) -> int:
        """Purge stored memories."""
        cat_enum = MemoryCategory(category.lower()) if category else None
        return self.store.clear(cat_enum)

    def consolidate(self, max_items: int = 500, purge_threshold: float = 0.08) -> int:
        """Garbage collection protocol: prune decayed low-importance items.

        CORE memories are permanently preserved.
        """
        all_items = self.store.list_all()
        if len(all_items) <= max_items:
            return 0

        purged = 0
        for item in all_items:
            if item.category == MemoryCategory.CORE:
                continue
            if item.compute_decay() < purge_threshold:
                self.store.delete(item.id)
                purged += 1
                if len(all_items) - purged <= max_items:
                    break

        return purged

    def get_stats(self) -> dict[str, Any]:
        """Return memory engine metrics and storage footprint."""
        all_items = self.store.list_all()
        by_category = {}
        for cat in MemoryCategory:
            by_category[cat.value] = sum(1 for i in all_items if i.category == cat)

        return {
            "total_memories": len(all_items),
            "by_category": by_category,
            "db_path": str(self.store.db_path.resolve()),
            "db_size_kb": round(self.store.size_bytes() / 1024, 2),
        }
