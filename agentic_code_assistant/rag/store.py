"""ChromaDB persistence. The index lives inside the *target* project (default `.aca/chroma`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from agentic_code_assistant.config import get_settings


@dataclass(frozen=True)
class IndexLocation:
    """Where an index lives: a Chroma directory plus a collection name."""

    persist_dir: Path
    collection_name: str

    @classmethod
    def for_project(cls, project_root: Path) -> IndexLocation:
        rag = get_settings().rag
        path = Path(rag.persist_dir)
        return cls(path if path.is_absolute() else project_root / path, rag.collection_name)


def get_collection(location: IndexLocation) -> Collection:
    location.persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(location.persist_dir))
    # cosine distance is the standard choice for text embeddings
    return client.get_or_create_collection(
        name=location.collection_name, metadata={"hnsw:space": "cosine"}
    )
