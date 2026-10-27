"""Unit tests for the mini-infer Agent Memory Engine."""

import time
from pathlib import Path

from mini_infer.memory.engine import MemoryEngine
from mini_infer.memory.retrieval import HybridRetriever
from mini_infer.memory.schema import MemoryCategory, MemoryItem
from mini_infer.memory.store import SQLiteMemoryStore


def test_memory_item_schema_and_decay():
    item = MemoryItem(content="Always use uv for package management", category=MemoryCategory.CORE)
    assert item.category == MemoryCategory.CORE
    assert item.compute_decay() == 1.0

    # Non-core item with simulated old access
    decaying_item = MemoryItem(
        content="Temporary scratchpad note",
        category=MemoryCategory.EPISODIC,
        last_accessed=time.time() - (86400 * 30),  # 30 days ago
    )
    decay = decaying_item.compute_decay()
    assert decay < 0.5  # Decayed substantially after 30 days

    # Serialization roundtrip
    d = item.to_dict()
    restored = MemoryItem.from_dict(d)
    assert restored.content == item.content
    assert restored.category == item.category


def test_sqlite_store_crud(tmp_path: Path):
    db_file = tmp_path / "test_store.db"
    store = SQLiteMemoryStore(db_file)

    assert store.count() == 0

    item1 = MemoryItem(
        content="Target GPU is RTX 3050 6GB", category=MemoryCategory.CORE, importance=1.5
    )
    item2 = MemoryItem(
        content="INT4 quantization uses group_size 64", category=MemoryCategory.PROJECT
    )

    store.insert(item1)
    store.insert(item2)

    assert store.count() == 2
    assert store.count(MemoryCategory.CORE) == 1
    assert store.count(MemoryCategory.PROJECT) == 1

    # Get by ID
    retrieved = store.get(item1.id)
    assert retrieved is not None
    assert retrieved.content == item1.content
    assert retrieved.importance == 1.5

    # Update access
    store.update_access(item1.id)
    updated = store.get(item1.id)
    assert updated.access_count == 1

    # Delete
    deleted = store.delete(item2.id)
    assert deleted is True
    assert store.count() == 1

    # Clear
    store.clear()
    assert store.count() == 0


def test_bm25_and_hybrid_retriever():
    items = [
        MemoryItem(
            content="Vectorized INT4 blockwise quantization with quantbench",
            category=MemoryCategory.PROJECT,
        ),
        MemoryItem(
            content="Continuous batching iteration scheduler and PagedAttention pool",
            category=MemoryCategory.PROJECT,
        ),
        MemoryItem(
            content="FastAPI streaming HTTP server on port 8000 using uvicorn",
            category=MemoryCategory.PROJECT,
        ),
    ]

    retriever = HybridRetriever(alpha=0.5)

    # 1. Search for quantization
    res_quant = retriever.rank("INT4 quantization algorithms", items, top_k=2)
    assert len(res_quant) > 0
    assert "quantization" in res_quant[0][0].content

    # 2. Search for batching / attention
    res_sched = retriever.rank("paged KV memory scheduler", items, top_k=2)
    assert len(res_sched) > 0
    assert "batching" in res_sched[0][0].content or "PagedAttention" in res_sched[0][0].content


def test_memory_engine_deduplication(tmp_path: Path):
    db_file = tmp_path / "test_engine.db"
    engine = MemoryEngine(db_file)

    # Add initial fact
    item1 = engine.add(
        "User preference: strictly write type-annotated Python code", category="core"
    )
    assert engine.store.count() == 1
    assert item1.access_count == 0

    # Add near-identical fact
    item2 = engine.add(
        "User preference: strictly write type annotated Python code", category="core"
    )
    assert engine.store.count() == 1  # Deduplicated!
    assert item2.id == item1.id
    assert item2.access_count >= 1


def test_memory_engine_prompt_formatting(tmp_path: Path):
    db_file = tmp_path / "test_prompt.db"
    engine = MemoryEngine(db_file)

    engine.add("Target GPU is RTX 3050 6GB with 192 GB/s bandwidth", category="core")
    engine.add("Project uses quantbench-engine for INT4 nibble packing", category="project")
    engine.add("Ignore unrelated weather forecast queries", category="episodic")

    prompt_ctx = engine.format_prompt_context(
        "What GPU memory optimizations are we using?", max_tokens=200
    )

    assert "[PERSISTENT AGENT MEMORY - VERIFIED FACTS]" in prompt_ctx
    assert "RTX 3050" in prompt_ctx or "quantbench-engine" in prompt_ctx
    assert "[END PERSISTENT MEMORY]" in prompt_ctx


def test_memory_engine_stats_and_consolidation(tmp_path: Path):
    db_file = tmp_path / "test_stats.db"
    engine = MemoryEngine(db_file)

    engine.add("Fact 1", category="core")
    engine.add("Fact 2", category="project")
    engine.add("Fact 3", category="decision")

    stats = engine.get_stats()
    assert stats["total_memories"] == 3
    assert stats["by_category"]["core"] == 1
    assert stats["by_category"]["project"] == 1
    assert stats["by_category"]["decision"] == 1
    assert stats["db_size_kb"] > 0


def test_memory_cli_commands(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    from mini_infer.cli import app

    test_db = tmp_path / "cli_mem.db"
    monkeypatch.setenv("MINI_INFER_MEMORY_DB", str(test_db))

    runner = CliRunner()

    # 1. Add
    res = runner.invoke(
        app,
        ["memory", "add", "CLI test memory item", "--category", "core", "--importance", "1.5"],
    )
    assert res.exit_code == 0
    assert "Memory Stored" in res.stdout

    # 2. List
    res = runner.invoke(app, ["memory", "list"])
    assert res.exit_code == 0
    assert "CLI test memory item" in res.stdout

    # 3. Search
    res = runner.invoke(app, ["memory", "search", "test memory"])
    assert res.exit_code == 0
    assert "CLI test memory item" in res.stdout

    # 4. Stats
    res = runner.invoke(app, ["memory", "stats"])
    assert res.exit_code == 0
    assert "Total Memories: 1" in res.stdout

    # 5. Clear
    res = runner.invoke(app, ["memory", "clear", "--force"])
    assert res.exit_code == 0
    assert "Purged 1 memory item(s)" in res.stdout
