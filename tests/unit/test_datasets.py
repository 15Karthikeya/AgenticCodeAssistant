"""Guard the hand-written goldens: wrong reference data would silently corrupt every eval."""

import json
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[1] / "evals"
FIXTURE = EVALS / "fixtures" / "sample_project"
RAG = json.loads((EVALS / "dataset_rag.json").read_text(encoding="utf-8"))
AGENT = json.loads((EVALS / "dataset_codebase_agent.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", RAG, ids=[r["input"][:40] for r in RAG])
def test_rag_golden_snippets_exist_verbatim_in_the_fixture(row):
    assert (FIXTURE / row["source_file"]).is_file()
    all_source = [p.read_text(encoding="utf-8") for p in FIXTURE.rglob("*.py")]
    assert row["expected_output"].strip() and row["input"].strip()
    for snippet in row["context"]:  # a golden may need evidence from more than one file
        assert any(snippet in text for text in all_source), "snippet not found in fixture"


def test_datasets_have_no_duplicate_questions_and_required_fields():
    for data in (RAG, AGENT):
        questions = [r["input"] for r in data]
        assert len(questions) == len(set(questions))
        assert all(r["input"] and r["expected_output"] for r in data)
    assert len(RAG) >= 15 and len(AGENT) >= 12
