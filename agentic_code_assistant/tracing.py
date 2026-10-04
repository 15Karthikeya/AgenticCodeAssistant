"""Optional DeepEval tracing.

`observe` records each decorated call as a span, so evals can see which tools the agent used
and in what order. It is OFF by default (a no-op decorator) so normal use never imports
deepeval or leaves files in your project. The eval suite turns it on by setting ACA_TRACING=1
(see tests/evals/__init__.py), before the app modules are imported.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _noop_observe(*args: Any, **kwargs: Any) -> Any:
    if args and callable(args[0]) and not kwargs:  # used as bare @observe
        return args[0]

    def decorator(fn: F) -> F:
        return fn

    return decorator


observe: Callable[..., Any] = _noop_observe

if os.getenv("ACA_TRACING") == "1":
    try:
        from deepeval.tracing import observe as observe  # noqa: F811
    except ImportError:  # deepeval not installed: stay a no-op
        pass
