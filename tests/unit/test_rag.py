from agentic_code_assistant.agent.tools import build_search_tool, format_results
from agentic_code_assistant.rag.indexer import index_codebase
from agentic_code_assistant.rag.retriever import RetrievedChunk, get_retriever
from agentic_code_assistant.rag.store import IndexLocation
from tests.unit.conftest import HashingEmbeddings

EMB = HashingEmbeddings()


def _loc(tmp_path):
    return IndexLocation(tmp_path / "chroma", "test")


def test_index_then_retrieve_finds_the_right_file(project, tmp_path):
    loc = _loc(tmp_path)
    stats = index_codebase(project, location=loc, embeddings=EMB)
    assert stats["indexed"] == stats["files"] > 0 and stats["skipped"] == 0

    retrieve = get_retriever(project_root=project, location=loc, embeddings=EMB)
    top = retrieve("ExpenseStore save fingerprint disk unchanged", k=3)
    assert len(top) == 3
    assert top[0].path == "expense_tracker/store.py"
    assert top[0].distance <= top[1].distance  # best match first


def test_second_run_skips_unchanged_files(project, tmp_path):
    loc = _loc(tmp_path)
    index_codebase(project, location=loc, embeddings=EMB)
    again = index_codebase(project, location=loc, embeddings=EMB)
    assert again["indexed"] == 0 and again["skipped"] == again["files"]


def test_changed_file_is_reindexed_and_deleted_file_is_removed(project, tmp_path):
    loc = _loc(tmp_path)
    index_codebase(project, location=loc, embeddings=EMB)

    (project / "expense_tracker" / "models.py").write_text("def brand_new_function():\n    pass\n")
    (project / "README.md").unlink()
    stats = index_codebase(project, location=loc, embeddings=EMB)
    assert stats["indexed"] == 1 and stats["deleted"] == 1

    retrieve = get_retriever(project_root=project, location=loc, embeddings=EMB)
    assert retrieve("brand_new_function", k=1)[0].path == "expense_tracker/models.py"
    assert all(c.path != "README.md" for c in retrieve("sample expense tracker", k=10))


def test_changing_chunk_settings_invalidates_old_entries(project, tmp_path, monkeypatch):
    from agentic_code_assistant.config import get_settings

    loc = _loc(tmp_path)
    index_codebase(project, location=loc, embeddings=EMB)
    monkeypatch.setattr(get_settings().rag, "chunk_lines", 30)
    assert index_codebase(project, location=loc, embeddings=EMB)["indexed"] > 0


def test_ignored_dirs_big_and_binary_files_are_not_indexed(project, tmp_path):
    (project / ".venv").mkdir()
    (project / ".venv" / "lib.py").write_text("def hidden():\n    pass\n")
    (project / "huge.json").write_text("x" * 300_000)
    (project / "bin.json").write_bytes(b"\x00\x01\x02binary")
    loc = _loc(tmp_path)
    index_codebase(project, location=loc, embeddings=EMB)
    paths = {c.path for c in get_retriever(project_root=project, location=loc, embeddings=EMB)("x", k=50)}
    assert not paths & {".venv/lib.py", "huge.json", "bin.json"}


def test_format_results_drops_whole_low_ranked_chunks_to_fit_budget():
    chunks = [RetrievedChunk(text="word " * 200, path=f"f{i}.py", start_line=1, end_line=9) for i in range(5)]
    out = format_results(chunks, max_tokens=500)
    assert "### f0.py:1-9" in out and "omitted" in out and "### f4.py" not in out


def test_search_tool_returns_error_text_when_retrieval_breaks(project, tmp_path):
    tool = build_search_tool(project, location=_loc(tmp_path), embeddings=EMB)
    # nothing indexed yet -> empty result, handled gracefully
    assert "No results" in tool.invoke({"query": "anything"})
