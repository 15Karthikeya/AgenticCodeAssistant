class ExpenseNotFoundError(KeyError):
    """Raised when delete/get is called with an unknown expense id."""


class BudgetExceededError(Exception):
    """Raised when adding an expense would push a category over its monthly limit."""

    def __init__(self, category: str, limit: float, attempted_total: float) -> None:
        self.category = category
        self.limit = limit
        self.attempted_total = attempted_total
        super().__init__(
            f"Category {category!r} would reach {attempted_total:.2f}, "
            f"exceeding monthly limit {limit:.2f}"
        )
