from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from agentic_code_assistant.config import get_settings, require_openai_key


def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=get_settings().embeddings.model,
        api_key=require_openai_key(),
    )
