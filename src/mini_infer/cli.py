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
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model path or HuggingFace repo ID"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    num_blocks: int = typer.Option(1024, "--num-blocks", help="Number of physical Paged KV blocks"),
    block_size: int = typer.Option(16, "--block-size", help="Tokens per physical block"),
    device: str = typer.Option("auto", "--device", help="Device (cuda/cpu/auto)"),
):
    """Launch the OpenAI-compatible streaming HTTP server."""
    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from mini_infer import server
    from mini_infer.engine import InferenceEngine
    from mini_infer.models.loader import load_model_from_safetensors

    selected_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    console.print(f"[bold cyan]mini-infer[/bold cyan] serving [bold green]{model}[/bold green] on [yellow]{host}:{port}[/yellow] ({selected_device})")

    # Load tokenizer
    console.print("Loading tokenizer...")
    try:
        server.tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load tokenizer ({e}). Using character fallback.[/yellow]")

    # Load model
    console.print("Loading model weights...")
    model_instance, config = load_model_from_safetensors(model, device=selected_device)

    # Initialize inference engine
    console.print(f"Initializing Paged KV-Cache ({num_blocks} blocks of size {block_size})...")
    server.engine = InferenceEngine(
        model=model_instance,
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        device=torch.device(selected_device),
    )

    console.print(f"[bold green]Server ready![/bold green] Connect any OpenAI-compatible client or run [bold]servebench quick[/bold] against http://{host}:{port}/v1")
    uvicorn.run(server.app, host=host, port=port, log_level="info")


@app.command()
def info():
    """Display system hardware and GPU VRAM budget for mini-infer."""
    import torch

    console.print("[bold cyan]mini-infer System Diagnostics[/bold cyan]")
    console.print(f"PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    console.print(f"CUDA available: [{'green' if cuda_avail else 'red'}]{cuda_avail}[/]")

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        console.print(f"GPU Device: [bold green]{device_name}[/bold green]")
        console.print(f"Total VRAM: [bold yellow]{total_vram_gb:.2f} GB[/bold yellow]")
        console.print("\n[bold]Memory Allocation Plan (RTX 3050 6 GB):[/bold]")
        console.print("  • Qwen2.5-1.5B (Target Model): ~3.0 GB VRAM")
        console.print("  • Qwen2.5-0.5B (Draft Model):  ~1.0 GB VRAM")
        console.print("  • Paged KV-Cache Memory Pool:  ~1.5 GB VRAM (~2000 blocks)")
        console.print("  • Safety Buffer:               ~0.5 GB")
