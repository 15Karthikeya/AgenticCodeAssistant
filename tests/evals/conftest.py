"""Index the fixture project ONCE per session into a throwaway Chroma directory.

The evals therefore never touch a real index, and never index the assistant's own source:
the assistant is always evaluated on a project it has not seen (the expense_tracker fixture).
"""

from __future__ import annotations

import pytest

from agentic_code_assistant.rag.indexer import index_codebase
from agentic_code_assistant.rag.store import IndexLocation
from tests.evals.metrics import FIXTURE_ROOT


@pytest.fixture(scope="session")
def eval_index(tmp_path_factory: pytest.TempPathFactory) -> IndexLocation:
    location = IndexLocation(tmp_path_factory.mktemp("eval_chroma"), "aca_eval_sample_project")
    index_codebase(FIXTURE_ROOT, location=location)
    return location
