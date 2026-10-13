"""Agent evals: run the real ReAct agent on a project it has never seen, then score the run.

`handle_query` and every tool are decorated with @observe, so DeepEval records a trace
(which tools were called, in what order, with what results). `assert_test(golden=...)` scores
that trace. Run with `deepeval test run`, not plain pytest, so the trace collector is active.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from agentic_code_assistant.agent.orchestrator import handle_query
from tests.evals.metrics import FIXTURE_ROOT, agent_metrics, load_dataset

GOLDENS = load_dataset("dataset_codebase_agent.json")


@pytest.mark.parametrize("row", GOLDENS, ids=[f"agent-{i:02d}" for i in range(len(GOLDENS))])
def test_codebase_agent(row: dict, eval_index, tmp_path: Path) -> None:
    # The agent has a shell tool, so give it a scratch copy: nothing it does can touch the fixture.
    scratch = tmp_path / "sample_project"
    shutil.copytree(FIXTURE_ROOT, scratch)

    handle_query(row["input"], project_root=scratch, location=eval_index)

    golden = Golden(input=row["input"], expected_output=row["expected_output"])
    assert_test(golden=golden, metrics=agent_metrics())
