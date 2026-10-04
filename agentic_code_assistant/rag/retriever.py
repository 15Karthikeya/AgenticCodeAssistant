"""Top-k retrieval. This one function is used by the `search_codebase` tool AND by the RAG evals,
so the evals measure exactly what the agent sees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from agentic_code_assistant.config import get_settings
from agentic_code_assistant.paths import project_root_from_cwd
from agentic_code_assistant.rag.embeddings import get_embeddings
from agentic_code_assistant.rag.store import IndexLocation, get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    path: str
    start_line: int
    end_line: int
    distance: float | None = None  # cosine distance: lower = more similar


Retriever = Callable[..., list[RetrievedChunk]]


def get_retriever(
    *,
    project_root: Path | None = None,
    location: IndexLocation | None = None,
    embeddings: Embeddings | None = None,
) -> Retriever:
    """Return `retrieve(query, k=None) -> list[RetrievedChunk]`, best match first."""
    root = (project_root or project_root_from_cwd()).resolve()
    embeddings = embeddings or get_embeddings()
    collection = get_collection(location or IndexLocation.for_project(root))
    default_k = get_settings().rag.top_k

    def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
        result = collection.query(
            query_embeddings=[embeddings.embed_query(query)],
            n_results=k or default_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                text=doc,
                path=str(meta.get("path", "")),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                distance=float(dist),
            )
            for doc, meta, dist in zip(documents, metadatas, distances, strict=False)
        ]

    return retrieve
