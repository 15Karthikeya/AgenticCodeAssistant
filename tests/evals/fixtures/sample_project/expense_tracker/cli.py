from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from expense_tracker.budget import BudgetChecker
from expense_tracker.exceptions import BudgetExceededError, ExpenseNotFoundError
from expense_tracker.store import ExpenseStore

DEFAULT_DB = Path("expenses.json")
DEFAULT_LIMITS = {"food": 300.0, "transport": 120.0, "fun": 80.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expense", description="Personal expense tracker CLI")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Record an expense if it fits the monthly category budget")
    add.add_argument("amount", type=float)
    add.add_argument("category")
    add.add_argument("--note", default="")
    add.add_argument("--date", dest="incurred_on", default=None)

    sub.add_parser("list", help="List expenses, newest first")

    delete = sub.add_parser("delete", help="Delete by id")
    delete.add_argument("expense_id")

    sub.add_parser("summary", help="Print category totals for the current month")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ExpenseStore(args.db)
    checker = BudgetChecker(DEFAULT_LIMITS)

    if args.command == "add":
        incurred = date.fromisoformat(args.incurred_on) if args.incurred_on else date.today()
        try:
            checker.ensure_within_budget(
                store, category=args.category, amount=args.amount, today=incurred
            )
        except BudgetExceededError as exc:
            print(exc)
            return 2
        expense = store.add(args.amount, args.category, args.note, incurred_on=incurred)
        store.save()
        print(f"added {expense.id} {expense.amount:.2f} {expense.category}")
        return 0

    if args.command == "list":
        for expense in store.list_all():
            print(f"{expense.id}\t{expense.incurred_on}\t{expense.amount:.2f}\t{expense.category}\t{expense.note}")
        return 0

    if args.command == "delete":
        try:
            removed = store.delete(args.expense_id)
        except ExpenseNotFoundError:
            print(f"unknown id: {args.expense_id}")
            return 1
        store.save()
        print(f"deleted {removed.id}")
        return 0

    if args.command == "summary":
        today = date.today()
        totals = store.totals_by_category(today.year, today.month)
        if not totals:
            print("no expenses this month")
            return 0
        for category, total in sorted(totals.items()):
            limit = checker.limit_for(category)
            extra = f" / {limit:.2f}" if limit is not None else ""
            print(f"{category}: {total:.2f}{extra}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
