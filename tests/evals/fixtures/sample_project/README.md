# Sample expense tracker

Tiny JSON-backed tracker used as the **eval fixture**. It is unrelated to the ACA agent.

- `ExpenseStore` persists to `expenses.json` and skips writes when the snapshot is unchanged
- `BudgetChecker.ensure_within_budget` raises `BudgetExceededError` when a category would exceed its monthly limit
- CLI: `python -m expense_tracker.cli add 12 food --note lunch`
