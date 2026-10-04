"""`search_codebase`: the RAG tool. Retrieval from ChromaDB, trimmed to fit the context window."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool

from agentic_code_assistant.config import get_settings
from agentic_code_assistant.context import count_tokens, trim_text
from agentic_code_assistant.rag.retriever import RetrievedChunk, get_retriever
from agentic_code_assistant.rag.store import IndexLocation
from agentic_code_assistant.tracing import observe


def format_results(chunks: list[RetrievedChunk], max_tokens: int) -> str:
    """Render chunks best-first and stop adding once the token budget is used up.

    Whole chunks are kept or dropped (never cut in half) so the model always sees complete code.
    """
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks):
        block = f"### {chunk.path}:{chunk.start_line}-{chunk.end_line}\n{chunk.text}"
        cost = count_tokens(block)
        if used + cost > max_tokens:
            if not parts:  # a single oversized chunk: keep a trimmed version rather than nothing
                parts.append(trim_text(block, max_tokens))
            omitted = len(chunks) - len(parts)
            parts.append(f"[{omitted} lower-ranked result(s) omitted to fit the context window]")
            break
        parts.append(block)
        used += cost
    return "\n\n".join(parts)


def build_search_tool(
    project_root: Path,
    *,
    location: IndexLocation | None = None,
    embeddings: Embeddings | None = None,
):
    retrieve = get_retriever(project_root=project_root, location=location, embeddings=embeddings)
    max_tokens = get_settings().agent.max_tool_output_tokens

    @tool
    @observe(type="tool", name="search_codebase")
    def search_codebase(query: str) -> str:
        """Semantic search over the project's code (RAG). Returns the most relevant code chunks
        as `path:start-end` plus the code. Describe what you are looking for in plain words or
        use identifiers (class / function names)."""
        try:
            chunks = retrieve(query)
        except Exception as exc:  # becomes an observation the agent can react to
            return f"search_codebase failed: {type(exc).__name__}: {exc}"
        if not chunks:
            return "No results. The index may be empty; try list_directory / read_file instead."
        return format_results(chunks, max_tokens)

    return search_codebase
