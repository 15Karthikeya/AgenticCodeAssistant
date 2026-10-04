from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentic_code_assistant.agent.graph import build_graph
from agentic_code_assistant.agent.orchestrator import handle_query
from agentic_code_assistant.rag.indexer import index_codebase
from agentic_code_assistant.rag.store import IndexLocation
from tests.unit.conftest import HashingEmbeddings, ScriptedChatModel, tool_call

EMB = HashingEmbeddings()


def _model(*script):
    return ScriptedChatModel(script=list(script), calls=[])


def test_react_loop_reasons_acts_observes_then_answers(project, tmp_path):
    loc = IndexLocation(tmp_path / "chroma", "test")
    index_codebase(project, location=loc, embeddings=EMB)
    llm = _model(
        tool_call("search_codebase", query="save fingerprint"),
        AIMessage(content="save() returns False when the fingerprint is unchanged."),
    )
    steps = []
    answer = handle_query(
        "How does save work?", project_root=project, location=loc, llm=llm, embeddings=EMB,
        on_step=lambda kind, text: steps.append(kind),
    )
    assert "fingerprint" in answer
    assert steps == ["act", "observe"]
    # the 2nd LLM call must have seen the tool observation (that is the "observe" in ReAct)
    assert any(isinstance(m, ToolMessage) and "store.py" in m.content for m in llm.calls[1])


def test_iteration_limit_forces_a_graceful_answer():
    @tool
    def ping(x: str) -> str:
        """ping"""
        return "pong"

    llm = _model(
        tool_call("ping", "1", x="a"),
        tool_call("ping", "2", x="b"),  # 2nd reasoning step hits the limit -> not executed
        AIMessage(content="Best effort answer."),  # produced by the finalize node
    )
    graph = build_graph(llm, [ping], system_prompt="sys", max_iterations=2)
    result = graph.invoke({"messages": [HumanMessage("q")], "steps": 0})
    assert result["messages"][-1].content == "Best effort answer."
    assert result["steps"] == 2  # never reasoned a 3rd time as an agent
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1  # only the first request was executed
    # finalize call: valid history (no dangling tool call) + the "answer now" nudge
    last_call = llm.calls[-1]
    assert isinstance(last_call[-1], HumanMessage) and "Answer now" in last_call[-1].content
    assert not (isinstance(last_call[-2], AIMessage) and last_call[-2].tool_calls)


def test_tool_exception_becomes_an_observation_and_agent_recovers():
    @tool
    def boom(x: str) -> str:
        """always fails"""
        raise RuntimeError("disk on fire")

    llm = _model(tool_call("boom", x="a"), AIMessage(content="Could not verify; tool failed."))
    graph = build_graph(llm, [boom], system_prompt="sys", max_iterations=5)
    result = graph.invoke({"messages": [HumanMessage("q")], "steps": 0})
    observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "disk on fire" in observation.content
    assert result["messages"][-1].content.startswith("Could not verify")


def test_unknown_tool_name_does_not_crash():
    @tool
    def real(x: str) -> str:
        """real"""
        return "ok"

    llm = _model(tool_call("made_up_tool", x="a"), AIMessage(content="done"))
    graph = build_graph(llm, [real], system_prompt="sys", max_iterations=5)
    result = graph.invoke({"messages": [HumanMessage("q")], "steps": 0})
    assert result["messages"][-1].content == "done"


def test_runtime_llm_failure_is_reported_not_raised(project, tmp_path):
    class Broken(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise ConnectionError("network down")

    llm = Broken(script=[AIMessage(content="x")], calls=[])
    out = handle_query(
        "q", project_root=project, location=IndexLocation(tmp_path / "c", "test"), llm=llm, embeddings=EMB
    )
    assert "failed" in out and "network down" in out
