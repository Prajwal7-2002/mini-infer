import typer
from rich.console import Console

app = typer.Typer(
    name="mini-infer",
    help="Minimal continuous-batching LLM inference engine with paged KV-cache.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model path or HuggingFace repo ID"
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    num_blocks: int = typer.Option(1024, "--num-blocks", help="Number of physical Paged KV blocks"),
    block_size: int = typer.Option(16, "--block-size", help="Tokens per physical block"),
    device: str = typer.Option("auto", "--device", help="Device (cuda/cpu/auto)"),
    kv_cache_dtype: str = typer.Option(
        "auto", "--kv-cache-dtype", help="KV-cache data type: auto or int8 (50% memory saving)"
    ),
    quant: str | None = typer.Option(
        None,
        "--quant",
        "-q",
        help="Sub-byte weight quantization scheme: int4, int8, ternary, sparse24 (via quantbench-engine)",
    ),
    group_size: int = typer.Option(
        64, "--group-size", help="Block size for blockwise quantization (default: 64)"
    ),
):
    """Launch the OpenAI-compatible streaming HTTP server."""
    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from mini_infer import server
    from mini_infer.engine import InferenceEngine
    from mini_infer.models.loader import load_model_from_safetensors

    selected_device = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )
    quant_info = f", quant={quant.upper()}" if quant else ""
    console.print(
        f"[bold cyan]mini-infer[/bold cyan] serving [bold green]{model}[/bold green] on [yellow]{host}:{port}[/yellow] ({selected_device}) [kv_cache={kv_cache_dtype}{quant_info}]"
    )

    # Load tokenizer
    console.print("Loading tokenizer...")
    try:
        server.tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not load tokenizer ({e}). Using character fallback.[/yellow]"
        )

    # Load model
    console.print("Loading model weights...")
    model_instance, config = load_model_from_safetensors(
        model, device=selected_device, quant_scheme=quant, group_size=group_size
    )

    # Initialize inference engine
    console.print(
        f"Initializing Paged KV-Cache ({num_blocks} blocks of size {block_size}, dtype={kv_cache_dtype})..."
    )
    server.engine = InferenceEngine(
        model=model_instance,
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        device=torch.device(selected_device),
        kv_cache_dtype=kv_cache_dtype,
    )

    console.print(
        f"[bold green]Server ready![/bold green] Connect any OpenAI-compatible client or run [bold]servebench quick[/bold] against http://{host}:{port}/v1"
    )
    uvicorn.run(server.app, host=host, port=port, log_level="info")


@app.command()
def repomap(
    path: str = typer.Argument(".", help="Directory path to index"),
    max_tokens: int = typer.Option(1500, "--max-tokens", "-t", help="Target token budget"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Optional output markdown file path"
    ),
):
    """Generate an AST-based repository map skeleton for LLM coding prompts."""
    from pathlib import Path

    from mini_infer.repomap import build_repo_map, estimate_tokens

    target_path = Path(path)
    if not target_path.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{path}' does not exist.")
        raise typer.Exit(1)

    console.print(
        f"Indexing repository AST for [cyan]{target_path.resolve()}[/cyan] (budget: {max_tokens} tokens)..."
    )
    skeleton = build_repo_map(target_path, max_tokens=max_tokens)
    est = estimate_tokens(skeleton)
    console.print(f"[bold green]Repo map generated:[/bold green] ~{est} tokens")

    if output:
        out_p = Path(output)
        out_p.write_text(skeleton, encoding="utf-8")
        console.print(f"Saved to [yellow]{out_p.resolve()}[/yellow]")
    else:
        console.print("\n" + skeleton)


@app.command()
def chat(
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model path or HuggingFace repo ID"
    ),
    context: str | None = typer.Option(
        None, "--context", "-c", help="Directory to index or markdown context file path"
    ),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Maximum tokens per completion"),
    device: str = typer.Option("auto", "--device", help="Device (cuda/cpu/auto)"),
    kv_cache_dtype: str = typer.Option(
        "int8", "--kv-cache-dtype", help="KV-cache precision (auto/int8)"
    ),
    quant: str | None = typer.Option(
        None,
        "--quant",
        "-q",
        help="Sub-byte weight quantization scheme: int4, int8, ternary, sparse24 (via quantbench-engine)",
    ),
    group_size: int = typer.Option(
        64, "--group-size", help="Block size for blockwise quantization (default: 64)"
    ),
    memory: bool = typer.Option(
        True,
        "--memory/--no-memory",
        help="Enable persistent Agent Memory recall across turns and sessions",
    ),
):
    """Interactive terminal coding chat with automatic codebase context, prefix caching, and agent memory."""
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer

    from mini_infer.engine import InferenceEngine
    from mini_infer.memory import MemoryEngine
    from mini_infer.models.loader import load_model_from_safetensors
    from mini_infer.repomap import build_repo_map, estimate_tokens

    selected_device = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )
    quant_info = f", quant={quant.upper()}" if quant else ""
    mem_info = ", memory=active" if memory else ""

    console.print(
        f"[bold cyan]mini-infer interactive chat[/bold cyan] ({selected_device}, kv={kv_cache_dtype}{quant_info}{mem_info})"
    )

    # Initialize Agent Memory Engine
    memory_engine = MemoryEngine() if memory else None
    if memory_engine:
        mem_stats = memory_engine.get_stats()
        console.print(
            f"[dim][Memory] Persistent facts loaded: {mem_stats['total_memories']} ({mem_stats['db_size_kb']} KB)[/dim]"
        )

    context_str = ""
    if context:
        c_path = Path(context)
        if c_path.is_dir():
            console.print(f"Indexing repository AST for [cyan]{c_path.resolve()}[/cyan]...")
            context_str = build_repo_map(c_path, max_tokens=1200)
            console.print(
                f"[green]Indexed codebase:[/green] ~{estimate_tokens(context_str)} tokens"
            )
        elif c_path.is_file():
            context_str = c_path.read_text(encoding="utf-8")
            console.print(
                f"[green]Loaded context from {c_path.name}:[/green] ~{estimate_tokens(context_str)} tokens"
            )
        else:
            console.print(
                f"[yellow]Warning: Context path '{context}' not found. Starting chat without codebase context.[/yellow]"
            )

    console.print(f"Loading weights for [bold green]{model}[/bold green]...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_instance, config = load_model_from_safetensors(
        model, device=selected_device, quant_scheme=quant, group_size=group_size
    )

    engine = InferenceEngine(
        model=model_instance,
        config=config,
        num_blocks=1024,
        block_size=16,
        device=torch.device(selected_device),
        kv_cache_dtype=kv_cache_dtype,
        enable_prefix_caching=True,
    )

    console.print(
        "\n[bold green]Chat ready![/bold green] Ask any question about your codebase (type '/remember <fact>', '/memory', 'exit' to stop).\n"
    )

    messages = []
    base_system = "You are an expert AI software engineer. Provide clean, production-ready code with concise explanations."
    if context_str:
        base_system += f"\n\nYou have access to this codebase outline:\n\n{context_str}"

    req_idx = 0
    while True:
        try:
            user_input = typer.prompt("User")
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.strip().lower() in ("exit", "quit", "q"):
            break
        if not user_input.strip():
            continue

        # In-chat memory management shortcuts
        if user_input.startswith("/remember "):
            fact = user_input[len("/remember ") :].strip()
            if memory_engine and fact:
                item = memory_engine.add(fact, category="core", importance=1.5)
                console.print(
                    f"[bold green][Memorized] ({item.category.value}):[/bold green] {item.content}\n"
                )
            continue

        if user_input.strip() == "/memory":
            if memory_engine:
                all_m = memory_engine.store.list_all()
                console.print(f"\n[bold cyan]Persistent Memories ({len(all_m)} items):[/bold cyan]")
                for m in all_m[:10]:
                    console.print(f"  - [{m.category.value}] {m.content}")
                console.print()
            continue

        # Construct system prompt with dynamic memory injection
        active_system = base_system
        if memory_engine:
            mem_block = memory_engine.format_prompt_context(user_input, max_tokens=250)
            if mem_block:
                active_system = f"{base_system}\n{mem_block}"

        messages.append({"role": "user", "content": user_input})

        # Inject updated system prompt as first message
        chat_payload = [{"role": "system", "content": active_system}] + [
            m for m in messages if m["role"] != "system"
        ]

        prompt_text = tokenizer.apply_chat_template(
            chat_payload, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text)

        req_id = f"chat-{req_idx}"
        req_idx += 1
        engine.add_request(prompt_ids, max_tokens=max_tokens, request_id=req_id)

        console.print("[bold cyan]mini-infer:[/bold cyan] ", end="")
        gen_tokens = []
        while engine.scheduler.has_unfinished_sequences():
            out = engine.step()
            if out:
                for s, tok_id in out:
                    if s.request_id == req_id:
                        gen_tokens.append(tok_id)
                        token_str = tokenizer.decode([tok_id], skip_special_tokens=True)
                        print(token_str, end="", flush=True)
        print("\n")
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": reply})


@app.command()
def info():
    """Display system hardware, GPU VRAM budget, and quantbench-engine compression profiles."""
    import torch
    from rich.table import Table

    console.print("[bold cyan]mini-infer System Diagnostics[/bold cyan]")
    console.print(f"PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    console.print(f"CUDA available: [{'green' if cuda_avail else 'red'}]{cuda_avail}[/]")

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"GPU Device: [bold green]{device_name}[/bold green]")
        console.print(f"Total VRAM: [bold yellow]{total_vram_gb:.2f} GB[/bold yellow]")

    # Precision & Quantization Memory Footprint Matrix (quantbench-engine integration)
    table = Table(
        title="Model Footprint by Quantization Scheme (quantbench-engine)",
        header_style="bold magenta",
    )
    table.add_column("Scheme", style="cyan", justify="left")
    table.add_column("Precision", justify="center")
    table.add_column("Qwen2.5-0.5B", justify="right")
    table.add_column("Qwen2.5-1.5B", justify="right")
    table.add_column("Max KV Blocks (6 GB VRAM)", justify="right", style="green")

    table.add_row("FP16 (Dense)", "16-bit Float", "0.98 GB", "3.08 GB", "~1,800 blocks")
    table.add_row("INT8 Absmax", "8-bit Signed", "0.51 GB", "1.58 GB", "~3,200 blocks")
    table.add_row("INT4 Packed", "4-bit Nibble", "0.28 GB", "0.84 GB", "~4,800 blocks")
    table.add_row("BitNet Ternary", "1.58-bit {-1,0,1}", "0.16 GB", "0.48 GB", "~5,600 blocks")
    table.add_row("Ampere 2:4 Sparse", "50% Structured", "0.54 GB", "1.65 GB", "~3,100 blocks")
    console.print()
    console.print(table)


# ==============================================================================
# Agent Memory Subcommands
# ==============================================================================
memory_app = typer.Typer(
    name="memory",
    help="Manage persistent Agent Memory (facts, preferences, decisions, and episodic context).",
    no_args_is_help=True,
)


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="Memory content or fact to store"),
    category: str = typer.Option(
        "project", "--category", "-c", help="Category: core, project, decision, episodic"
    ),
    importance: float = typer.Option(
        1.0, "--importance", "-i", help="Importance weight (0.1 to 2.0)"
    ),
):
    """Add a verified fact or cognitive memory."""
    from mini_infer.memory import MemoryCategory, MemoryEngine

    engine = MemoryEngine()
    try:
        cat = MemoryCategory(category.lower())
    except ValueError:
        console.print(
            f"[bold red]Error:[/bold red] Invalid category '{category}'. Choose from: core, project, decision, episodic."
        )
        raise typer.Exit(1) from None

    item = engine.add(content, category=cat, importance=importance)
    console.print(
        f"[bold green][Memory Stored]:[/bold green] [{item.category.value}] {item.content} (id: {item.id})"
    )


@memory_app.command("list")
def memory_list(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter by category (core, project, decision, episodic)"
    ),
    limit: int = typer.Option(25, "--limit", "-n", help="Maximum items to display"),
):
    """List persistent agent memories."""
    from rich.table import Table

    from mini_infer.memory import MemoryCategory, MemoryEngine

    engine = MemoryEngine()
    cat_filter = None
    if category:
        try:
            cat_filter = MemoryCategory(category.lower())
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Invalid category '{category}'.")
            raise typer.Exit(1) from None

    items = engine.store.list_all(category=cat_filter)[:limit]
    if not items:
        console.print("[dim]No memories found in store.[/dim]")
        return

    table = Table(title=f"Persistent Agent Memories ({len(items)} items)", header_style="bold cyan")
    table.add_column("ID", style="dim", justify="left")
    table.add_column("Category", style="magenta", justify="center")
    table.add_column("Content", style="green", justify="left")
    table.add_column("Imp", justify="center")
    table.add_column("Hits", justify="center")
    table.add_column("Decay", justify="right")

    for item in items:
        decay = item.compute_decay()
        table.add_row(
            item.id,
            item.category.value,
            item.content,
            f"{item.importance:.1f}",
            str(item.access_count),
            f"{decay:.2f}",
        )

    console.print(table)


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Query or phrase to retrieve memories for"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    threshold: float = typer.Option(0.10, "--threshold", "-t", help="Minimum relevance threshold"),
):
    """Search agent memories using hybrid BM25 + dense subword feature ranking."""
    from rich.table import Table

    from mini_infer.memory import MemoryEngine

    engine = MemoryEngine()
    results = engine.search(query, top_k=top_k, threshold=threshold)
    if not results:
        console.print(f"[yellow]No matching memories found for query: '{query}'[/yellow]")
        return

    table = Table(title=f"Hybrid Memory Search: '{query}'", header_style="bold green")
    table.add_column("Score", style="bold yellow", justify="right")
    table.add_column("Category", style="magenta", justify="center")
    table.add_column("Content", style="cyan", justify="left")
    table.add_column("Accesses", justify="center")

    for item, score in results:
        table.add_row(
            f"{score:.4f}",
            item.category.value,
            item.content,
            str(item.access_count),
        )

    console.print(table)


@memory_app.command("stats")
def memory_stats():
    """Display memory database size, category distribution, and storage path."""
    from rich.table import Table

    from mini_infer.memory import MemoryEngine

    engine = MemoryEngine()
    stats = engine.get_stats()

    console.print("[bold cyan]Agent Memory Engine Telemetry[/bold cyan]")
    console.print(f"Store Path: [yellow]{stats['db_path']}[/yellow]")
    console.print(f"Database Size: [bold green]{stats['db_size_kb']} KB[/bold green]")
    console.print(f"Total Memories: [bold]{stats['total_memories']}[/bold]\n")

    table = Table(title="Memory Distribution by Category", header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for cat, count in stats["by_category"].items():
        table.add_row(cat.capitalize(), str(count))

    console.print(table)


@memory_app.command("clear")
def memory_clear(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Category to purge (omit to clear entire memory)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt"),
):
    """Purge memories from persistent storage."""
    from mini_infer.memory import MemoryEngine

    engine = MemoryEngine()
    target = f"category '{category}'" if category else "ALL persistent memories"
    if not force:
        confirm = typer.confirm(f"Are you sure you want to permanently clear {target}?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            return

    purged = engine.clear(category=category)
    console.print(f"[bold red]Purged {purged} memory item(s).[/bold red]")


@memory_app.command("consolidate")
def memory_consolidate(
    max_items: int = typer.Option(500, "--max-items", help="Maximum memories to retain"),
):
    """Run cognitive garbage collection, pruning decayed episodic items while preserving core rules."""
    from mini_infer.memory import MemoryEngine

    engine = MemoryEngine()
    pruned = engine.consolidate(max_items=max_items)
    console.print(
        f"[bold green]Consolidation complete:[/bold green] pruned {pruned} decayed item(s)."
    )


app.add_typer(memory_app, name="memory")


if __name__ == "__main__":
    app()
