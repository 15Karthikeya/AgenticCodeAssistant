"""Offline fakes so the real pipeline runs without an API key."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "sample_project"


class HashingEmbeddings(Embeddings):
    """Bag-of-words embedding: identical words -> overlapping vectors. Deterministic, free."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for word in re.findall(r"[a-z_]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per LLM call. Records what it was sent."""

    script: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence, **kwargs):  # tool schemas are irrelevant here
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        reply = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, call_id: str = "c1", **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A writable copy of the fixture project."""
    import shutil

    dest = tmp_path / "proj"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest
