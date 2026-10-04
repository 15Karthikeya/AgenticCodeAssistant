"""The ReAct loop as an explicit LangGraph state machine.

            +----------------------------+
   START -> |  reason  (LLM + tools)     | -- no tool calls -----------------> END
            +----------------------------+
               |  tool calls, budget left        ^
               v                                 |
            +----------------------------+       |
            |  act  (run the tools)      | ------+   (observations go back to the LLM)
            +----------------------------+
   reason -- tool calls but step budget used up --> finalize (answer from what we have) -> END

Why build it by hand instead of using a prebuilt agent? The loop is only ~60 lines, every
piece (state, nodes, routing, stop rule) is visible and testable, and the iteration limit
becomes a graceful "answer with what you have" instead of an exception.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

REASON, ACT, FINALIZE = "reason", "act", "finalize"

STEP_LIMIT_NUDGE = (
    "You have used all of your allowed tool steps. Do not call any more tools. "
    "Answer now using only what you have gathered, and say clearly what you could not verify."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # whole conversation, appended to
    steps: int  # how many times the LLM has been asked to reason


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    *,
    system_prompt: str,
    max_iterations: int,
):
    llm_with_tools = llm.bind_tools(tools)

    def reason(state: AgentState) -> dict:
        """Thought: the LLM reads the conversation and either answers or requests tool calls."""
        response = llm_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END  # plain answer: done
        if state["steps"] >= max_iterations:
            return FINALIZE  # iteration limit: stop acting, force an answer
        return ACT

    def finalize(state: AgentState) -> dict:
        """Out of budget: drop the unanswerable tool request and ask for a best-effort answer."""
        history = state["messages"][:-1]  # last message is an AI turn whose tool calls we skip
        response = llm.invoke(
            [SystemMessage(system_prompt), *history, HumanMessage(STEP_LIMIT_NUDGE)]
        )
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node(REASON, reason)
    # handle_tool_errors=True: if a tool raises, the error text becomes the observation
    # and the agent can recover, instead of the whole run crashing.
    graph.add_node(ACT, ToolNode(tools, handle_tool_errors=True))
    graph.add_node(FINALIZE, finalize)
    graph.add_edge(START, REASON)
    graph.add_conditional_edges(REASON, route, {ACT: ACT, FINALIZE: FINALIZE, END: END})
    graph.add_edge(ACT, REASON)
    graph.add_edge(FINALIZE, END)
    return graph.compile()
