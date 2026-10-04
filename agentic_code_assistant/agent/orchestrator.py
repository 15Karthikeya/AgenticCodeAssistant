"""`handle_query`: the single entry point used by the CLI and by the evals."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agentic_code_assistant.agent.factory import build_agent
from agentic_code_assistant.config import get_settings
from agentic_code_assistant.paths import project_root_from_cwd
from agentic_code_assistant.rag.store import IndexLocation
from agentic_code_assistant.tracing import observe

# on_step("act" | "observe", text): lets the CLI show the agent's progress live
StepCallback = Callable[[str, str], None]


def _text(content: object) -> str:
    """Message content is a string, or a list of blocks for some providers."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else str(b.get("text", "")) for b in content if b
        )
    return str(content)


def _report(message: BaseMessage, on_step: StepCallback | None) -> str | None:
    """Forward progress to the callback; return the text if this is a final answer."""
    if isinstance(message, AIMessage):
        if message.tool_calls:
            if on_step:
                for call in message.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    on_step("act", f"{call['name']}({args})")
            return None
        return _text(message.content)
    if isinstance(message, ToolMessage) and on_step:
        on_step("observe", _text(message.content))
    return None


@observe(type="agent", name="handle_query")
def handle_query(
    query: str,
    *,
    project_root: Path | None = None,
    location: IndexLocation | None = None,
    on_step: StepCallback | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
) -> str:
    """Run the ReAct agent on one question and return its final answer as text.

    Configuration problems (e.g. missing API key) raise immediately. Problems that happen
    *while the agent runs* (API outage, step limit) are turned into a readable message.
    """
    settings = get_settings()
    root = (project_root or project_root_from_cwd()).resolve()
    agent = build_agent(root, location=location, llm=llm, embeddings=embeddings)

    # Backstop only: the graph stops itself at max_iterations (see graph.route). Each loop
    # is two graph steps (reason + act), so allow a little more than twice that.
    config = {"recursion_limit": 2 * settings.agent.max_iterations + 5}

    answer = ""
    try:
        for event in agent.stream(
            {"messages": [HumanMessage(query)], "steps": 0}, config, stream_mode="updates"
        ):
            for update in event.values():
                for message in update.get("messages", []):
                    answer = _report(message, on_step) or answer
    except GraphRecursionError:
        return "Stopped: the agent exceeded its safety limit. Try a more specific question."
    except Exception as exc:  # API outage, rate limit after retries, ...
        return f"The agent failed: {type(exc).__name__}: {exc}"
    return answer or "The agent finished without producing an answer."
