"""Command line interface: `aca index` and `aca ask "..."`, run from inside *another* project."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from agentic_code_assistant import __version__
from agentic_code_assistant.paths import project_root_from_cwd

app = typer.Typer(
    help="Ask questions about the project in the current directory (ReAct agent + RAG).",
    no_args_is_help=True,
)
console = Console()


def _index(root) -> dict[str, int]:
    from agentic_code_assistant.rag.indexer import index_codebase

    return index_codebase(root)


@app.command()
def index() -> None:
    """Build or refresh the vector index of the current directory (incremental)."""
    root = project_root_from_cwd()
    console.print(f"[bold]Indexing[/bold] {root}")
    try:
        stats = _index(root)
    except RuntimeError as exc:  # e.g. missing OPENAI_API_KEY
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"{stats['files']} files seen: {stats['indexed']} (re)indexed, "
        f"{stats['skipped']} unchanged, {stats['deleted']} removed"
    )


def _show_step(kind: str, text: str) -> None:
    if kind == "act":
        console.print(f"[magenta]act[/magenta]      {escape(text)}")
    else:
        console.print(f"[dim]observe  {escape(' '.join(text.split())[:160])}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about this codebase"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide the agent's tool steps"),
    reindex: bool = typer.Option(True, "--reindex/--no-reindex", help="Refresh the index first"),
) -> None:
    """Answer a question about the project in the current directory."""
    from agentic_code_assistant.agent.orchestrator import handle_query

    root = project_root_from_cwd()
    try:
        if reindex:
            stats = _index(root)  # cheap when nothing changed: unchanged files are only hashed
            if stats["indexed"] or stats["deleted"]:
                console.print(f"[dim]index updated: {stats['indexed']} file(s) (re)indexed[/dim]")
        answer = handle_query(
            question, project_root=root, on_step=None if quiet else _show_step
        )
    except RuntimeError as exc:  # e.g. missing OPENAI_API_KEY
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(Panel(Markdown(answer), title="Answer", border_style="cyan"))


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)
