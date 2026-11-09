"""Независимая библиотека генерации и кластерного анализа нечётких величин.

Расстояние — RMS-различие двух концов единственного α-среза.
Алгоритм — иерархическая агломеративная кластеризация, complete linkage.
Останов: число кластеров k ИЛИ максимальный допустимый межкластерный порог.
"""

from dataclasses import dataclass
import heapq
import numpy as np
from fuzzy_numbers import FuzzyNumber, exponential, DEFAULT_ALPHA


@dataclass(frozen=True)
class Sample:
    a: float
    lam: float
    number: FuzzyNumber


def generate_samples(n: int, seed: int = 42, mode: str = "groups",
                     alpha: float = DEFAULT_ALPHA) -> list[Sample]:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 100 <= n <= 300:
        raise ValueError("Количество нечётких величин должно быть от 100 до 300.")
    rng = np.random.default_rng(seed)
    if mode == 'groups':
        center_a = np.array([1.0, 4.15, 7.3, 10.45])
        center_lam = np.array([0.65, 1.65, 0.9, 1.9])
        cls = rng.integers(0, len(center_a), size=n)
        aa = np.clip(rng.normal(center_a[cls], 0.42), 0, 12)
        lam = np.clip(rng.normal(center_lam[cls], 0.14), 0.2, 2.5)
    elif mode == 'uniform':
        aa = rng.uniform(0.0, 12.0, n)
        lam = rng.uniform(0.35, 2.4, n)
    else:
        raise ValueError("Режим генерации: groups или uniform.")
    return [Sample(float(a), float(p), exponential(float(a), float(p), f"X{i+1}", alpha=alpha))
            for i, (a, p) in enumerate(zip(aa, lam))]


def pairwise_distances(numbers: list[FuzzyNumber]) -> np.ndarray:
    if not numbers:
        raise ValueError("Набор нечётких величин пуст.")
    ref = numbers[0].alphas
    if any(not np.array_equal(num.alphas, ref) for num in numbers):
        raise ValueError("Все величины должны иметь один общий уровень α.")
    features = np.stack([np.concatenate((v.left, v.right)) for v in numbers])
    dist = np.zeros((len(numbers), len(numbers)))
    for i, feature in enumerate(features[:-1]):
        differences = features[i + 1:] - feature
        distances = np.sqrt(np.mean(differences**2, axis=1))
        dist[i, i + 1:] = distances
        dist[i + 1:, i] = distances
    return dist


def cluster_numbers(numbers: list[FuzzyNumber], *, k: int | None = None,
                    threshold: float | None = None) -> np.ndarray:
    """Complete linkage по RMS-расстоянию концов выбранных α-срезов.

    Ровно один параметр: k — итоговое число кластеров; либо threshold —
    максимальное расстояние между элементами объединяемых кластеров.
    """
    if (k is None) == (threshold is None):
        raise ValueError("Укажите либо k, либо порог, но не оба параметра.")
    n = len(numbers)
    if not n:
        raise ValueError("Нечего кластеризовать.")
    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n):
        raise ValueError(f"Количество кластеров k должно быть в диапазоне 1..{n}.")
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("Порог должен быть положительным конечным числом.")

    initial = pairwise_distances(numbers)
    active = set(range(n))
    members = {i: [i] for i in range(n)}
    distances = {}
    heap = []
    for i in range(n):
        for j in range(i+1, n):
            d = float(initial[i, j])
            distances[(i, j)] = d
            heap.append((d, i, j))
    heapq.heapify(heap)
    next_id = n
    while len(active) > 1 and (k is None or len(active) > k):
        while heap:
            dist, i, j = heapq.heappop(heap)
            if i in active and j in active:
                break
        else:
            break
        if threshold is not None and dist > threshold:
            break
        active.remove(i)
        active.remove(j)
        current = next_id
        next_id += 1
        members[current] = members.pop(i) + members.pop(j)
        # Lance–Williams для complete linkage: D(A∪B,C)=max(D(A,C),D(B,C)).
        for other in active:
            key1 = (min(i, other), max(i, other))
            key2 = (min(j, other), max(j, other))
            new_d = max(distances[key1], distances[key2])
            key_new = (min(other, current), max(other, current))
            distances[key_new] = new_d
            heapq.heappush(heap, (new_d, *key_new))
        active.add(current)
    # Для воспроизводимого отображения сортируем кластеры по средней левой границе.
    ordered = sorted(active, key=lambda group: float(np.mean(
        [numbers[idx].left[-1] for idx in members[group]])))
    labels = np.empty(n, dtype=int)
    for label, group in enumerate(ordered, start=1):
        labels[members[group]] = label
    return labels
