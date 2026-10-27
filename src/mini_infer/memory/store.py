"""SQLite persistent storage provider for Agent Memory."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from mini_infer.memory.schema import MemoryCategory, MemoryItem


class SQLiteMemoryStore:
    """ACID-compliant embedded SQLite storage for agent memory items."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        elif os.getenv("MINI_INFER_MEMORY_DB"):
            self.db_path = Path(os.environ["MINI_INFER_MEMORY_DB"])
        else:
            base_dir = Path(".mini_infer")
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = base_dir / "memory.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create memory schema if table does not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);")

    def insert(self, item: MemoryItem) -> None:
        """Insert or replace a memory item."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, category, content, importance, created_at,
                    last_accessed, access_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    item.id,
                    item.category.value,
                    item.content,
                    item.importance,
                    item.created_at,
                    item.last_accessed,
                    item.access_count,
                    json.dumps(item.metadata),
                ),
            )

    def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve memory item by unique ID."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM memories WHERE id = ?;", (item_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_item(row)

    def list_all(self, category: MemoryCategory | None = None) -> list[MemoryItem]:
        """Fetch all stored memories, optionally filtered by category."""
        with self._get_connection() as conn:
            if category is not None:
                cur = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY last_accessed DESC;",
                    (category.value,),
                )
            else:
                cur = conn.execute("SELECT * FROM memories ORDER BY last_accessed DESC;")
            return [self._row_to_item(row) for row in cur.fetchall()]

    def update_access(self, item_id: str, timestamp: float | None = None) -> None:
        """Record an access event, updating last_accessed and incrementing access_count."""
        import time

        now = timestamp if timestamp is not None else time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?;
                """,
                (now, item_id),
            )

    def delete(self, item_id: str) -> bool:
        """Delete a memory item by ID."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?;", (item_id,))
            return cur.rowcount > 0

    def clear(self, category: MemoryCategory | None = None) -> int:
        """Purge stored memories."""
        with self._get_connection() as conn:
            if category is not None:
                cur = conn.execute("DELETE FROM memories WHERE category = ?;", (category.value,))
            else:
                cur = conn.execute("DELETE FROM memories;")
            return cur.rowcount

    def count(self, category: MemoryCategory | None = None) -> int:
        """Count total stored memory items."""
        with self._get_connection() as conn:
            if category is not None:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE category = ?;", (category.value,)
                )
            else:
                cur = conn.execute("SELECT COUNT(*) FROM memories;")
            return int(cur.fetchone()[0])

    def size_bytes(self) -> int:
        """Return file size of SQLite database in bytes."""
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            category=MemoryCategory(row["category"]),
            content=row["content"],
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )
