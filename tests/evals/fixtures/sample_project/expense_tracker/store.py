from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from expense_tracker.exceptions import ExpenseNotFoundError
from expense_tracker.models import Expense


class ExpenseStore:
    """JSON-backed store. `save()` is a no-op when the in-memory snapshot hash matches disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._expenses: dict[str, Expense] = {}
        self._disk_fingerprint = ""
        if self.path.exists():
            self._load()

    def add(self, amount: float, category: str, note: str, incurred_on: date | None = None) -> Expense:
        expense = Expense(
            id=uuid4().hex[:8],
            amount=amount,
            category=category.lower().strip(),
            note=note,
            incurred_on=incurred_on or date.today(),
        )
        self._expenses[expense.id] = expense
        return expense

    def get(self, expense_id: str) -> Expense:
        try:
            return self._expenses[expense_id]
        except KeyError as exc:
            raise ExpenseNotFoundError(expense_id) from exc

    def delete(self, expense_id: str) -> Expense:
        expense = self.get(expense_id)
        del self._expenses[expense_id]
        return expense

    def list_all(self) -> list[Expense]:
        return sorted(self._expenses.values(), key=lambda e: e.incurred_on, reverse=True)

    def for_month(self, year: int, month: int, category: str | None = None) -> list[Expense]:
        items = [
            e
            for e in self._expenses.values()
            if e.incurred_on.year == year and e.incurred_on.month == month
        ]
        if category:
            needle = category.lower().strip()
            items = [e for e in items if e.category == needle]
        return items

    def totals_by_category(self, year: int, month: int) -> dict[str, float]:
        totals: dict[str, float] = {}
        for expense in self.for_month(year, month):
            totals[expense.category] = totals.get(expense.category, 0.0) + expense.amount
        return totals

    def save(self) -> bool:
        """Persist to JSON. Returns False when contents are unchanged vs last load/save."""
        payload = self._fingerprint()
        if payload == self._disk_fingerprint:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._to_json(), indent=2), encoding="utf-8")
        self._disk_fingerprint = payload
        return True

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._expenses = {
            item["id"]: Expense(
                id=item["id"],
                amount=float(item["amount"]),
                category=item["category"],
                note=item.get("note", ""),
                incurred_on=date.fromisoformat(item["incurred_on"]),
            )
            for item in raw.get("expenses", [])
        }
        self._disk_fingerprint = self._fingerprint()

    def _to_json(self) -> dict:
        return {
            "expenses": [
                {
                    "id": e.id,
                    "amount": e.amount,
                    "category": e.category,
                    "note": e.note,
                    "incurred_on": e.incurred_on.isoformat(),
                }
                for e in self.list_all()
            ]
        }

    def _fingerprint(self) -> str:
        return json.dumps(self._to_json(), sort_keys=True)
