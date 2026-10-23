import tempfile
from pathlib import Path

from mini_infer.repomap import build_repo_map, estimate_tokens, extract_file_skeleton


def test_extract_file_skeleton():
    sample_code = '''"""Sample module docstring."""

from typing import Optional

CONSTANT_VAL = 42

class DemoService:
    """Service class for demo purposes."""

    def __init__(self, host: str, port: int = 8080) -> None:
        self.host = host
        self.port = port

    def run(self, timeout: Optional[float] = 5.0) -> bool:
        """Run the service loop."""
        return True

    def _private_helper(self) -> None:
        pass


def global_compute(x: int, y: int = 10) -> int:
    """Compute something."""
    return x + y
'''

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(sample_code)
        temp_path = Path(f.name)

    try:
        # Public skeleton
        skel = extract_file_skeleton(temp_path, include_docstrings=True, include_private=False)
        assert "class DemoService:" in skel
        assert "def __init__(self, host: str, port: int = 8080) -> None: ..." in skel
        assert "def run(self, timeout: Optional[float] = 5.0) -> bool: ..." in skel
        assert "def global_compute(x: int, y: int = 10) -> int: ..." in skel
        assert "_private_helper" not in skel
        assert "Sample module docstring." in skel
    finally:
        temp_path.unlink(missing_ok=True)


def test_build_repo_map_budget_and_filtering():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        # File 1 in main
        (root / "main.py").write_text(
            "class MainApp:\n    def start(self) -> None: pass\n", encoding="utf-8"
        )

        # File 2 in subfolder
        sub = root / "pkg"
        sub.mkdir()
        (sub / "util.py").write_text(
            "def helper(data: list[str]) -> int:\n    return len(data)\n", encoding="utf-8"
        )

        # Ignored folder
        ignored = root / ".venv" / "lib"
        ignored.mkdir(parents=True)
        (ignored / "vendor.py").write_text("def vendor_func(): pass\n", encoding="utf-8")

        # Build repo map
        map_out = build_repo_map(root, max_tokens=1000)

        assert "main.py" in map_out
        assert "class MainApp:" in map_out
        assert "util.py" in map_out
        assert "def helper(data: list[str]) -> int: ..." in map_out
        assert "vendor_func" not in map_out
        assert estimate_tokens(map_out) <= 1000
