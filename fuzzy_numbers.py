"""Независимая библиотека нечётких величин на основе α-срезов.

Для экспоненциального варианта μ(x) = exp(-λ(x-a)), x>=a, и 0 иначе.
Плотность экспоненциального распределения f(x)=λ exp(-λ(x-a)) нормирована
на свой максимум, чтобы получилась функция принадлежности [0, 1].
"""

from dataclasses import dataclass
import numpy as np

ALPHAS = np.linspace(0.01, 1.0, 199)


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
        if a.ndim != 1 or l.shape != a.shape or r.shape != a.shape or len(a) < 2:
            raise ValueError("α-срезы должны быть одномерными массивами одинаковой длины.")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(l)) and np.all(np.isfinite(r))):
            raise ValueError("α-срезы должны состоять из конечных чисел.")
        if a[0] <= 0 or a[-1] > 1 or np.any(np.diff(a) <= 0):
            raise ValueError("Уровни α должны строго возрастать и лежать в (0, 1].")
        if np.any(l > r + 1e-9):
            raise ValueError("Левая граница α-среза не может быть правее правой.")
        if np.any(np.diff(l) < -1e-8) or np.any(np.diff(r) > 1e-8):
            raise ValueError("α-срезы должны быть вложенными.")
        object.__setattr__(self, 'alphas', a)
        object.__setattr__(self, 'left', l)
        object.__setattr__(self, 'right', r)

    @property
    def representative(self) -> float:
        """Скаляр для условного сравнения: среднее середин α-срезов."""
        middle = (self.left + self.right) / 2.0
        return float(np.sum((middle[1:] + middle[:-1]) *
                    np.diff(self.alphas) / 2) / (self.alphas[-1] - self.alphas[0]))

    def membership(self, x) -> np.ndarray:
        """Численное восстановление μ(x) из α-срезов; ниже α_min считается 0."""
        x = np.asarray(x, dtype=float)
        flat = x.reshape(-1)
        inside = (flat[np.newaxis, :] >= self.left[:, np.newaxis]) & (
            flat[np.newaxis, :] <= self.right[:, np.newaxis])
        result = np.max(np.where(inside, self.alphas[:, np.newaxis], 0.0), axis=0)
        return result.reshape(x.shape)


def exponential(a: float, lam: float, name: str = "A", alphas=None) -> FuzzyNumber:
    """Экспоненциальная нечёткая величина со сдвигом a и интенсивностью λ>0.

    μ(x)=0 при x<a; μ(x)=exp(-λ(x-a)) при x>=a.
    α-срез: [a, a-ln(α)/λ], для α>0.
    """
    if not np.isfinite(a) or not np.isfinite(lam) or lam <= 0:
        raise ValueError("Параметр a должен быть конечным, λ должен быть > 0.")
    levels = ALPHAS.copy() if alphas is None else np.asarray(alphas, dtype=float)
    return FuzzyNumber(levels, np.full_like(levels, float(a)),
                       a - np.log(levels) / lam, name)


def exponential_membership(x, a: float, lam: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= a, np.exp(-lam * np.maximum(x - a, 0)), 0.0)


def calculate(first: FuzzyNumber, second: FuzzyNumber, operation: str) -> FuzzyNumber:
    """Интервальная арифметика по принципу обобщения Заде (α-срезы)."""
    if not np.array_equal(first.alphas, second.alphas):
        raise ValueError("Для операции нужны одинаковые уровни α.")
    l1, r1, l2, r2 = first.left, first.right, second.left, second.right
    if operation == '+':
        left, right = l1 + l2, r1 + r2
    elif operation == '-':
        left, right = l1 - r2, r1 - l2
    elif operation in ('*', '/'):
        if operation == '/':
            if np.any((l2 <= 0) & (r2 >= 0)):
                raise ValueError("Деление невозможно: α-срез делителя включает 0.")
            l2, r2 = 1.0 / r2, 1.0 / l2
        combinations = np.stack((l1*l2, l1*r2, r1*l2, r1*r2), axis=0)
        left, right = combinations.min(axis=0), combinations.max(axis=0)
    else:
        raise ValueError(f"Неизвестная операция: {operation}")
    return FuzzyNumber(first.alphas, left, right,
                       f"({first.name} {operation} {second.name})")


def compare(first: FuzzyNumber, second: FuzzyNumber, atol: float = 1e-8) -> str:
    """Условный порядок по скалярной характеристике (не строгий нечёткий порядок)."""
    diff = first.representative - second.representative
    if abs(diff) <= atol:
        return "≈"
    return "<" if diff < 0 else ">"
