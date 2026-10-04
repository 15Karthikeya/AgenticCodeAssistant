"""Assemble the agent: LLM + least-privilege toolset + ReAct graph."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agentic_code_assistant.agent.graph import build_graph
from agentic_code_assistant.agent.prompts import SYSTEM_PROMPT
from agentic_code_assistant.agent.tools import build_search_tool
from agentic_code_assistant.config import get_settings, require_openai_key
from agentic_code_assistant.paths import project_root_from_cwd
from agentic_code_assistant.rag.store import IndexLocation
from agentic_code_assistant.tools.filesystem_tools import build_filesystem_tools
from agentic_code_assistant.tools.terminal_tools import build_terminal_tool


def build_llm() -> ChatOpenAI:
    settings = get_settings().llm
    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        timeout=settings.timeout_seconds,
        max_retries=2,  # transient API errors / rate limits are retried automatically
        api_key=require_openai_key(),
    )


def build_agent(
    project_root: Path | None = None,
    *,
    location: IndexLocation | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
):
    """Build the compiled ReAct graph for one project.

    `llm` / `embeddings` can be injected, which is how the offline unit tests run the real
    graph with scripted fakes.
    """
    root = (project_root or project_root_from_cwd()).resolve()
    tools = [
        build_search_tool(root, location=location, embeddings=embeddings),
        *build_filesystem_tools(root),
        build_terminal_tool(root),
    ]
    return build_graph(
        llm or build_llm(),
        tools,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=get_settings().agent.max_iterations,
    )
