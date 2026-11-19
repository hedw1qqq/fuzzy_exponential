"""Генерация интервалов, фильтрация по α и кластеризация k-medoids."""

from math import log, isfinite

import numpy as np

from fuzzy_numbers import FuzzyNumber, exponential_membership


METRICS = (
    'Хаусдорф для границ',
    'Разность ширин',
    'Разность квадратов ширин',
    'Евклидово расстояние между границами',
)


def generate_samples(n: int, lam: float = 1.0, *, seed: int = 42) -> list[FuzzyNumber]:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 100 <= n <= 300:
        raise ValueError("Количество величин должно быть от 100 до 300.")
    if not isfinite(lam) or lam <= 0:
        raise ValueError("Интенсивность λ должна быть положительной и конечной.")
    rng = np.random.Generator(np.random.MT19937(seed))
    scale = 1 / lam
    # Как для ветви Вейбулла с k = 1, размер окна задаёт медиана распределения.
    median = log(2) * scale
    limit = max(1.0, 3 * median)
    result = []
    for _ in range(n):
        center = rng.exponential(scale)
        while center > limit:
            center = rng.exponential(scale)
        half_width = max(0.05, median * rng.uniform(0.03, 0.18))
        result.append(FuzzyNumber(max(0.0, center - half_width), center + half_width))
    return result


def alpha_mask(numbers: list[FuzzyNumber], lam: float, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("Уровень α должен лежать в [0, 1].")
    return exponential_membership([number.center for number in numbers], lam) >= alpha


def pairwise_distances(numbers: list[FuzzyNumber], metric: int = 0) -> np.ndarray:
    if not numbers:
        raise ValueError("Набор нечётких величин пуст.")
    left = np.array([number.left for number in numbers])
    right = np.array([number.right for number in numbers])
    if metric == 0:
        return np.maximum(np.abs(left[:, None] - left), np.abs(right[:, None] - right))
    if metric == 3:
        return np.hypot(left[:, None] - left, right[:, None] - right)
    width = right - left
    if metric == 2:
        width = width**2
    elif metric != 1:
        raise ValueError("Неизвестная метрика.")
    return np.abs(width[:, None] - width)


def k_medoids(distances: np.ndarray, k: int, max_iter: int = 100,
              seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть метки с нуля и индексы медоидов."""
    n = len(distances)
    if not 1 <= k <= n:
        raise ValueError(f"Количество кластеров должно быть от 1 до {n}.")
    if max_iter < 1:
        raise ValueError("Число итераций должно быть положительным.")
    rng = np.random.Generator(np.random.MT19937(seed))
    medoids = [int(rng.integers(n))]
    while len(medoids) < k:
        nearest = distances[:, medoids].min(axis=1)
        nearest[medoids] = -1
        medoids.append(int(nearest.argmax()))

    labels = np.full(n, -1)
    for _ in range(max_iter):
        new_labels = distances[:, medoids].argmin(axis=1)
        changed = not np.array_equal(labels, new_labels)
        labels = new_labels
        for group, medoid in enumerate(medoids):
            members = np.flatnonzero(labels == group)
            if members.size:
                costs = distances[np.ix_(members, members)].sum(axis=1)
                best = int(members[costs.argmin()])
            else:
                nearest = distances[:, medoids].min(axis=1)
                nearest[medoids] = -1
                best = int(nearest.argmax()) if len(medoids) < n else medoid
            if best != medoid:
                medoids[group] = best
                changed = True
        if not changed:
            break
    return labels, np.array(medoids)
