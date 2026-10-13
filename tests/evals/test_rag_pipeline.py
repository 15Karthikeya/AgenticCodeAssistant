"""RAG evals: precision@k and recall@k of the retriever. Retrieval only, no generation.

Calls the same `retrieve()` the `search_codebase` tool uses. A failing score here therefore
points at chunking / k / embeddings / ranking, never at the LLM or the agent's tool choices.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from agentic_code_assistant.rag.retriever import get_retriever
from tests.evals.metrics import FIXTURE_ROOT, load_dataset, rag_metrics

GOLDENS = load_dataset("dataset_rag.json")


@pytest.mark.parametrize("row", GOLDENS, ids=[f"rag-{i:02d}" for i in range(len(GOLDENS))])
def test_rag_retrieval(row: dict, eval_index) -> None:
    retrieve = get_retriever(project_root=FIXTURE_ROOT, location=eval_index)
    chunks = retrieve(row["input"])  # top-k from the config (default 5)

    test_case = LLMTestCase(
        input=row["input"],
        expected_output=row["expected_output"],  # reference answer, hand-written
        retrieval_context=[chunk.text for chunk in chunks],  # what the retriever returned
    )
    assert_test(test_case, rag_metrics())
