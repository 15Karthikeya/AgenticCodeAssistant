from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Expense:
    id: str
    amount: float
    category: str
    note: str
    incurred_on: date

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        if not self.category.strip():
            raise ValueError("category is required")
