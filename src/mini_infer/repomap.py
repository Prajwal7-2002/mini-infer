"""AST-based repository mapping utility for large-codebase context injection.

Extracts classes, function signatures, type annotations, and docstring outlines
from Python codebases into a compact token-budgeted skeleton suitable for LLM system prompts
and prefix caching.
"""

from __future__ import annotations

import ast
from pathlib import Path

DEFAULT_IGNORES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "site",
    "dist",
    "build",
    "node_modules",
    "egg-info",
    ".egg-info",
}


def _format_arg(arg: ast.arg) -> str:
    """Format an AST function argument with its type annotation if available."""
    if arg.annotation:
        try:
            return f"{arg.arg}: {ast.unparse(arg.annotation)}"
        except Exception:
            return arg.arg
    return arg.arg


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool = False) -> str:
    """Format function name, typed arguments, and return annotation."""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    args_list: list[str] = []

    # Position-only args
    if hasattr(node.args, "posonlyargs") and node.args.posonlyargs:
        args_list.extend(_format_arg(a) for a in node.args.posonlyargs)
        args_list.append("/")

    # Positional & keyword args
    num_args = len(node.args.args)
    num_defaults = len(node.args.defaults)
    first_default_idx = num_args - num_defaults

    for i, a in enumerate(node.args.args):
        formatted = _format_arg(a)
        if i >= first_default_idx:
            def_val = node.args.defaults[i - first_default_idx]
            try:
                formatted += f" = {ast.unparse(def_val)}"
            except Exception:
                pass
        args_list.append(formatted)

    # Vararg (*args)
    if node.args.vararg:
        args_list.append(f"*{_format_arg(node.args.vararg)}")

    # Kwonly args
    if node.args.kwonlyargs:
        if not node.args.vararg:
            args_list.append("*")
        for i, a in enumerate(node.args.kwonlyargs):
            formatted = _format_arg(a)
            if i < len(node.args.kw_defaults) and node.args.kw_defaults[i] is not None:
                try:
                    formatted += f" = {ast.unparse(node.args.kw_defaults[i])}"
                except Exception:
                    pass
            args_list.append(formatted)

    # Kwarg (**kwargs)
    if node.args.kwarg:
        args_list.append(f"**{_format_arg(node.args.kwarg)}")

    sig = f"{prefix}{node.name}({', '.join(args_list)})"
    if node.returns:
        try:
            sig += f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return sig


def extract_file_skeleton(file_path: Path | str, include_docstrings: bool = True, include_private: bool = False) -> str:
    """Parse a Python source file and extract classes, methods, and function signatures.

    Args:
        file_path: Path to the .py source file.
        include_docstrings: Whether to include 1-line docstring summaries.
        include_private: Whether to include private helper functions (_func).

    Returns:
        Formatted Python skeleton string.
    """
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except Exception as e:
        return f"# Could not parse {path.name}: {e}"

    lines: list[str] = []

    # Optional module docstring
    if include_docstrings:
        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            first_line = mod_doc.strip().split("\n")[0]
            lines.append(f'"""{first_line}"""')

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not include_private and node.name.startswith("_") and not node.name.startswith("__"):
                continue

            # Class header with base classes
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            bases_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"class {node.name}{bases_str}:")

            class_doc = ast.get_docstring(node)
            if include_docstrings and class_doc:
                first_line = class_doc.strip().split("\n")[0]
                lines.append(f'    """{first_line}"""')

            method_found = False
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not include_private and item.name.startswith("_") and not item.name.startswith("__"):
                        continue
                    method_sig = _format_function_signature(item, is_method=True)
                    lines.append(f"    {method_sig}: ...")
                    method_found = True

            if not method_found and (not class_doc or not include_docstrings):
                lines.append("    ...")

            lines.append("")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not include_private and node.name.startswith("_") and not node.name.startswith("__"):
                continue
            sig = _format_function_signature(node, is_method=False)
            lines.append(f"{sig}: ...")
            if include_docstrings:
                doc = ast.get_docstring(node)
                if doc:
                    first_line = doc.strip().split("\n")[0]
                    lines.append(f'    """{first_line}"""')

    return "\n".join(lines).strip()


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count (~3.8 characters per token for source code)."""
    return max(1, int(len(text) / 3.8))


def build_repo_map(
    root_dir: str | Path = ".",
    max_tokens: int = 1500,
    file_extensions: tuple[str, ...] = (".py",),
    include_docstrings: bool = True,
    include_private: bool = False,
) -> str:
    """Build a compact, token-budgeted repository skeleton map for LLM prompts.

    Args:
        root_dir: Root directory of the codebase.
        max_tokens: Maximum target token budget.
        file_extensions: File extensions to index (default: ('.py',)).
        include_docstrings: Whether to include 1-line docstring summaries.
        include_private: Whether to include private methods/helpers.

    Returns:
        Structured Markdown string containing the repository skeleton.
    """
    root = Path(root_dir).resolve()
    all_files: list[Path] = []

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in file_extensions:
            # Check if any parent part is in ignores
            rel_parts = path.relative_to(root).parts
            if any(part in DEFAULT_IGNORES or part.startswith(".") for part in rel_parts[:-1]):
                continue
            all_files.append(path)

    # First pass: full skeletons
    file_skeletons: list[tuple[str, str]] = []
    for f in all_files:
        rel_path = f.relative_to(root).as_posix()
        skeleton = extract_file_skeleton(f, include_docstrings=include_docstrings, include_private=include_private)
        if skeleton:
            file_skeletons.append((rel_path, skeleton))

    def render_output(skeletons: list[tuple[str, str]]) -> str:
        header = f"# Repository Map: {root.name}\n\n"
        sections = []
        for rel_path, skel in skeletons:
            sections.append(f"### `{rel_path}`\n```python\n{skel}\n```")
        return header + "\n\n".join(sections)

    output = render_output(file_skeletons)
    if estimate_tokens(output) <= max_tokens:
        return output

    # Second pass: strip docstrings if exceeding budget
    file_skeletons = [
        (rel_path, extract_file_skeleton(root / rel_path, include_docstrings=False, include_private=False))
        for rel_path, _ in file_skeletons
    ]
    output = render_output(file_skeletons)
    if estimate_tokens(output) <= max_tokens:
        return output

    # Third pass: truncate files to strictly fit within budget
    budget_skeletons: list[tuple[str, str]] = []
    accumulated_tokens = estimate_tokens(f"# Repository Map: {root.name}\n\n")

    for rel_path, skel in file_skeletons:
        block = f"### `{rel_path}`\n```python\n{skel}\n```\n\n"
        block_tokens = estimate_tokens(block)
        if accumulated_tokens + block_tokens > max_tokens:
            budget_skeletons.append(
                ("...", f"# ... remaining files omitted to fit {max_tokens} token budget ...")
            )
            break
        budget_skeletons.append((rel_path, skel))
        accumulated_tokens += block_tokens

    return render_output(budget_skeletons)
