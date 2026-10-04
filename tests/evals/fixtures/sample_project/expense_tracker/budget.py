from __future__ import annotations

from datetime import date

from expense_tracker.exceptions import BudgetExceededError
from expense_tracker.store import ExpenseStore


class BudgetChecker:
    """Enforces per-category monthly ceilings before an expense is committed."""

    def __init__(self, limits: dict[str, float]) -> None:
        self._limits = {k.lower().strip(): float(v) for k, v in limits.items()}

    def limit_for(self, category: str) -> float | None:
        return self._limits.get(category.lower().strip())

    def remaining(self, store: ExpenseStore, category: str, today: date | None = None) -> float | None:
        limit = self.limit_for(category)
        if limit is None:
            return None
        day = today or date.today()
        spent = store.totals_by_category(day.year, day.month).get(category.lower().strip(), 0.0)
        return limit - spent

    def ensure_within_budget(
        self,
        store: ExpenseStore,
        *,
        category: str,
        amount: float,
        today: date | None = None,
    ) -> None:
        limit = self.limit_for(category)
        if limit is None:
            return
        day = today or date.today()
        spent = store.totals_by_category(day.year, day.month).get(category.lower().strip(), 0.0)
        attempted = spent + amount
        if attempted > limit:
            raise BudgetExceededError(category.lower().strip(), limit, attempted)
