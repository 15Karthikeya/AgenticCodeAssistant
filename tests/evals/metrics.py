"""Shared dataset loader and metric factories for both eval suites."""

from __future__ import annotations

import json
import os
from pathlib import Path

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
    TaskCompletionMetric,
)
from deepeval.test_case import SingleTurnParams

EVALS_DIR = Path(__file__).resolve().parent
FIXTURE_ROOT = EVALS_DIR / "fixtures" / "sample_project"

# The "judge" LLM that scores outputs. Override: export DEEPEVAL_EVAL_MODEL=gpt-4o
JUDGE_MODEL = os.getenv("DEEPEVAL_EVAL_MODEL", "gpt-4o-mini")
THRESHOLD = 0.5


def load_dataset(name: str) -> list[dict]:
    return json.loads((EVALS_DIR / name).read_text(encoding="utf-8"))


def rag_metrics() -> list:
    """Retrieval quality only (no generation involved).

    Precision@k: are the relevant chunks ranked near the top of the k results?
    Recall@k:    does the top-k contain everything needed to produce the expected answer?
    """
    return [
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL),
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL),
    ]


def agent_metrics() -> list:
    """End-to-end agent quality, scored from the recorded trace of one run."""
    return [
        # Did the agent actually accomplish what the user asked? (judged from the whole trace)
        TaskCompletionMetric(threshold=THRESHOLD, model=JUDGE_MODEL),
        # Is the final answer consistent with the hand-written reference answer?
        GEval(
            name="Codebase Answer Correctness",
            criteria=(
                "Judge whether the actual output answers the input question correctly for the "
                "expense_tracker project, using the expected output as ground truth. Penalize "
                "wrong facts, invented functions or files, and claims not supported by the "
                "expected output. Do not penalize extra correct detail or different wording."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
        ),
    ]
