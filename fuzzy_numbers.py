"""Операции с интервалами и экспоненциальная функция принадлежности."""

from dataclasses import dataclass
from math import isfinite

import numpy as np


ALPHA = 0.8


@dataclass(frozen=True)
class FuzzyNumber:
    left: float
    right: float

    def __post_init__(self):
        if not (isfinite(self.left) and isfinite(self.right) and self.left <= self.right):
            raise ValueError("Границы интервала должны быть конечными и упорядоченными.")

    @property
    def center(self) -> float:
        return self.left / 2 + self.right / 2


def exponential_membership(x, lam: float) -> np.ndarray:
    if not isfinite(lam) or lam <= 0:
        raise ValueError("Интенсивность λ должна быть положительной и конечной.")
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.exp(-lam * np.maximum(x, 0)), 0.0)


def calculate(first: FuzzyNumber, second: FuzzyNumber, operation: str) -> FuzzyNumber:
    l1, r1, l2, r2 = first.left, first.right, second.left, second.right
    if operation == '+':
        left, right = l1 + l2, r1 + r2
    elif operation == '-':
        left, right = l1 - r2, r1 - l2
    elif operation in ('*', '/'):
        if operation == '/':
            if l2 <= 0 <= r2:
                raise ValueError("Деление невозможно: интервал делителя включает 0.")
            values = (l1 / l2, l1 / r2, r1 / l2, r1 / r2)
        else:
            values = (l1 * l2, l1 * r2, r1 * l2, r1 * r2)
        left, right = min(values), max(values)
    else:
        raise ValueError(f"Неизвестная операция: {operation}")
    return FuzzyNumber(left, right)


def compare(first: FuzzyNumber, second: FuzzyNumber) -> str:
    """Вернуть знак порядка непересекающихся интервалов или ≈ при пересечении."""
    if first.right < second.left:
        return '<'
    if first.left > second.right:
        return '>'
    return '≈'
