"""Учебная модель: одна нечёткая величина = один α-срез (интервал).

Функция принадлежности исходной экспоненциальной величины известна,
но результат интервальной арифметики при одном α не задаёт полной
функции принадлежности результата.
"""
from dataclasses import dataclass
import numpy as np

DEFAULT_ALPHA = 0.5


@dataclass(frozen=True)
class FuzzyNumber:
    alphas: np.ndarray
    left: np.ndarray
    right: np.ndarray
    name: str = "Нечёткая величина"

    def __post_init__(self):
        a = np.asarray(self.alphas, dtype=float).copy()
        l = np.asarray(self.left, dtype=float).copy()
        r = np.asarray(self.right, dtype=float).copy()
        if a.shape != (1,) or l.shape != (1,) or r.shape != (1,):
            raise ValueError("Одна величина должна содержать ровно один α-срез.")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(l)) and np.all(np.isfinite(r))):
            raise ValueError("Параметры α-среза должны быть конечными.")
        if not 0 < a[0] <= 1:
            raise ValueError("Уровень α должен лежать в (0, 1].")
        if l[0] > r[0]:
            raise ValueError("Левая граница интервала не может быть больше правой.")
        object.__setattr__(self, 'alphas', a)
        object.__setattr__(self, 'left', l)
        object.__setattr__(self, 'right', r)

    @property
    def alpha(self) -> float:
        return float(self.alphas[0])

    @property
    def representative(self) -> float:
        """Учебная числовая оценка — середина одного интервала."""
        return float((self.left[0] + self.right[0]) / 2)


def exponential(a: float, lam: float, name: str = "A",
                alpha: float = DEFAULT_ALPHA) -> FuzzyNumber:
    """Интервал исходной экспоненциальной величины на одном уровне α.

    μ(x)=0 для x<a, exp(-λ(x-a)) для x>=a.
    A_α=[a, a-ln(α)/λ], λ>0, 0<α<=1.
    """
    if not np.isfinite(a) or not np.isfinite(lam) or lam <= 0:
        raise ValueError("Параметр a должен быть конечным, λ должен быть > 0.")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Уровень α должен лежать в (0, 1].")
    return FuzzyNumber(np.array([alpha]), np.array([a]),
                       np.array([a - np.log(alpha) / lam]), name)


def exponential_membership(x, a: float, lam: float) -> np.ndarray:
    """Точная функция принадлежности только исходной величины."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= a, np.exp(-lam * np.maximum(x - a, 0)), 0.0)


def calculate(first: FuzzyNumber, second: FuzzyNumber, operation: str) -> FuzzyNumber:
    """Интервальная арифметика над срезами одного и того же уровня α."""
    if not np.array_equal(first.alphas, second.alphas):
        raise ValueError("Операции требуют одинакового уровня α у двух величин.")
    l1, r1, l2, r2 = first.left, first.right, second.left, second.right
    if operation == '+':
        left, right = l1 + l2, r1 + r2
    elif operation == '-':
        left, right = l1 - r2, r1 - l2
    elif operation in ('*', '/'):
        if operation == '/':
            if np.any((l2 <= 0) & (r2 >= 0)):
                raise ValueError("Деление невозможно: интервал делителя содержит 0.")
            l2, r2 = 1.0 / r2, 1.0 / l2
        combinations = np.stack((l1 * l2, l1 * r2, r1 * l2, r1 * r2))
        left, right = combinations.min(axis=0), combinations.max(axis=0)
    else:
        raise ValueError(f"Неизвестная операция: {operation}")
    return FuzzyNumber(first.alphas, left, right,
                       f"({first.name} {operation} {second.name})")


def compare(first: FuzzyNumber, second: FuzzyNumber, atol: float = 1e-8) -> str:
    """Сравнение середин интервалов при одном α, не полный нечёткий порядок."""
    if not np.array_equal(first.alphas, second.alphas):
        raise ValueError("Сравнение требует одинакового уровня α.")
    diff = first.representative - second.representative
    if abs(diff) <= atol:
        return "≈"
    return "<" if diff < 0 else ">"
