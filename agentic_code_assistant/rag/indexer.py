"""Walk a project, chunk its files, embed the chunks, store them in ChromaDB.

Indexing is incremental: each stored chunk remembers a hash of the file it came from.
On the next run, unchanged files are skipped (no embedding cost), changed files are
re-indexed, and files that no longer exist are removed from the index.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

from agentic_code_assistant.config import RAGSettings, get_settings
from agentic_code_assistant.paths import is_ignored_dir, project_root_from_cwd
from agentic_code_assistant.rag.chunker import CodeChunk, chunk_source
from agentic_code_assistant.rag.embeddings import get_embeddings
from agentic_code_assistant.rag.store import IndexLocation, get_collection

CHUNKER_VERSION = "1"  # bump when chunking logic changes so old indexes are rebuilt
EMBED_BATCH_SIZE = 64
SKIP_FILENAMES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock"}


def iter_source_files(root: Path, rag: RAGSettings) -> list[Path]:
    extensions = {e.lower() for e in rag.index_extensions}
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if name in SKIP_FILENAMES or path.suffix.lower() not in extensions:
                continue
            try:
                if path.stat().st_size > rag.max_file_bytes:
                    continue
            except OSError:
                continue
            files.append(path)
    return files


def _settings_signature(rag: RAGSettings, embedding_model: str) -> str:
    """Anything that changes chunk contents or vectors must invalidate old entries."""
    return f"{CHUNKER_VERSION}|{rag.chunk_lines}|{rag.chunk_overlap}|{embedding_model}"


def _file_digest(data: bytes, signature: str) -> str:
    h = hashlib.sha256()
    h.update(signature.encode())
    h.update(data)
    return h.hexdigest()


def _stored_digests(collection) -> dict[str, str]:
    """Map of relative file path -> digest recorded when it was last indexed."""
    digests: dict[str, str] = {}
    for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
        if meta and meta.get("path") and meta.get("file_hash"):
            digests[meta["path"]] = meta["file_hash"]
    return digests


def _delete_file(collection, rel_path: str) -> None:
    collection.delete(where={"path": rel_path})


def index_codebase(
    project_root: Path | None = None,
    *,
    location: IndexLocation | None = None,
    embeddings: Embeddings | None = None,
) -> dict[str, int]:
    """Index `project_root` (default: the current directory) and return simple stats."""
    settings = get_settings()
    root = (project_root or project_root_from_cwd()).resolve()
    location = location or IndexLocation.for_project(root)
    embeddings = embeddings or get_embeddings()  # fail fast (e.g. no API key) before touching disk
    collection = get_collection(location)
    signature = _settings_signature(settings.rag, settings.embeddings.model)

    known = _stored_digests(collection)
    files = iter_source_files(root, settings.rag)

    skipped = indexed = 0
    live: set[str] = set()
    pending: list[tuple[CodeChunk, str]] = []

    for path in files:
        rel = path.relative_to(root).as_posix()
        live.add(rel)
        data = path.read_bytes()
        if b"\x00" in data[:4096]:  # binary file with a text extension
            continue
        digest = _file_digest(data, signature)
        if known.get(rel) == digest:
            skipped += 1
            continue
        if rel in known:
            _delete_file(collection, rel)
        chunks = chunk_source(
            data.decode("utf-8", errors="replace"),
            path=rel,
            chunk_lines=settings.rag.chunk_lines,
            overlap=settings.rag.chunk_overlap,
        )
        pending.extend((chunk, digest) for chunk in chunks)
        indexed += 1

    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        _upsert(collection, embeddings, pending[start : start + EMBED_BATCH_SIZE])

    stale = set(known) - live
    for rel in stale:
        _delete_file(collection, rel)

    return {"files": len(files), "indexed": indexed, "skipped": skipped, "deleted": len(stale)}


def _upsert(collection, embeddings: Embeddings, batch: list[tuple[CodeChunk, str]]) -> None:
    documents = [chunk.text for chunk, _ in batch]
    collection.upsert(
        ids=[f"{c.path}:{c.start_line}-{c.end_line}" for c, _ in batch],
        documents=documents,
        embeddings=embeddings.embed_documents(documents),
        metadatas=[
            {
                "path": c.path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol": c.symbol,
                "file_hash": digest,
            }
            for c, digest in batch
        ],
    )
