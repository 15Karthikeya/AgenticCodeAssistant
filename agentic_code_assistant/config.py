"""Typed settings loaded from config.yaml, with a few environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Looks for a .env next to / above this package (i.e. the ACA repo), then the real environment.
load_dotenv()

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class LLMSettings(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    timeout_seconds: int = 60


class EmbeddingSettings(BaseModel):
    model: str = "text-embedding-3-small"


class AgentSettings(BaseModel):
    max_iterations: int = Field(default=10, ge=1)
    max_tool_output_tokens: int = Field(default=3000, ge=100)


class RAGSettings(BaseModel):
    persist_dir: str = ".aca/chroma"
    collection_name: str = "codebase"
    top_k: int = Field(default=5, ge=1)
    chunk_lines: int = Field(default=60, ge=5)
    chunk_overlap: int = Field(default=10, ge=0)
    max_file_bytes: int = 200_000
    index_extensions: list[str] = [".py", ".md", ".toml", ".yaml", ".yml", ".json"]

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> RAGSettings:
        if self.chunk_overlap >= self.chunk_lines:
            raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_lines")
        return self


class TerminalSettings(BaseModel):
    timeout_seconds: int = Field(default=30, ge=1)


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    agent: AgentSettings = AgentSettings()
    rag: RAGSettings = RAGSettings()
    terminal: TerminalSettings = TerminalSettings()


def load_settings(path: Path | None = None) -> Settings:
    path = path or Path(os.getenv("ACA_CONFIG", DEFAULT_CONFIG_PATH))
    raw = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)
    if model := os.getenv("OPENAI_MODEL"):
        settings.llm.model = model
    if emb := os.getenv("OPENAI_EMBEDDING_MODEL"):
        settings.embeddings.model = emb
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def require_openai_key() -> str:
    """Fail fast with a clear message instead of a stack trace deep inside LangChain."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
