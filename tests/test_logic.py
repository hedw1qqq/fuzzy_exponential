"""Проверка модели «одна величина — один α-срез» и кластеризации."""
import unittest
import numpy as np

from fuzzy_numbers import FuzzyNumber, exponential, calculate, compare, DEFAULT_ALPHA
from clustering import generate_samples, pairwise_distances, cluster_numbers


class FuzzyNumbersTest(unittest.TestCase):
    def test_one_alpha_cut(self):
        a = exponential(2, 0.5, alpha=0.5)
        self.assertEqual(len(a.alphas), 1)
        self.assertEqual(a.alpha, 0.5)
        self.assertAlmostEqual(a.left[0], 2)
        self.assertAlmostEqual(a.right[0], 2 - np.log(0.5) / 0.5)

    def test_reject_multiple_cuts(self):
        with self.assertRaisesRegex(ValueError, 'один α-срез'):
            FuzzyNumber(np.array([0.5, 1.0]), np.array([2, 2]), np.array([3, 2]))

    def test_alpha_one_is_singleton(self):
        a = exponential(2, 0.5, alpha=1)
        self.assertAlmostEqual(a.left[0], a.right[0])

    def test_arithmetic(self):
        a, b = exponential(2, 1, alpha=1), exponential(3, 2, alpha=1)
        self.assertAlmostEqual(calculate(a, b, '+').left[0], 5)
        self.assertAlmostEqual(calculate(a, b, '-').right[0], -1)
        self.assertAlmostEqual(calculate(a, b, '*').left[0], 6)
        self.assertAlmostEqual(calculate(a, b, '/').right[0], 2 / 3)

    def test_intervals_at_half_alpha(self):
        a, b = exponential(2, 1), exponential(3, 2)
        c = calculate(a, b, '+')
        self.assertAlmostEqual(c.left[0], 5)
        self.assertAlmostEqual(c.right[0], 5 + np.log(2) * 1.5)
        self.assertEqual(len(c.alphas), 1)
        self.assertAlmostEqual(c.representative, sum((c.left[0], c.right[0])) / 2)

    def test_compare(self):
        self.assertEqual(compare(exponential(1, 1), exponential(4, 1)), '<')

    def test_invalid_division(self):
        with self.assertRaisesRegex(ValueError, '0'):
            calculate(exponential(3, 1), exponential(0, 2), '/')

    def test_negative_intervals(self):
        a, b = exponential(-4, 1), exponential(0.5, 2)
        result = calculate(a, b, '*')
        self.assertLessEqual(result.left[0], result.right[0])

    def test_mismatched_alpha_rejected(self):
        a = exponential(2, 1, alpha=0.4)
        b = exponential(3, 1, alpha=0.5)
        with self.assertRaisesRegex(ValueError, 'α'):
            calculate(a, b, '+')
        with self.assertRaisesRegex(ValueError, 'α'):
            compare(a, b)
        with self.assertRaisesRegex(ValueError, 'α'):
            pairwise_distances([a, b])

    def test_invalid_alpha(self):
        for alpha in (-0.2, 0, 1.2, np.nan):
            with self.assertRaises(ValueError):
                exponential(2, 1, alpha=alpha)
        self.assertEqual(exponential(2, 1).alpha, DEFAULT_ALPHA)


class ClusteringTest(unittest.TestCase):
    def test_distance_at_large_shift(self):
        numbers = [exponential(1e8, 1), exponential(1e8 + 1, 1)]
        self.assertAlmostEqual(pairwise_distances(numbers)[0, 1], 1.0)

    def test_generation_and_k(self):
        samples = generate_samples(100, alpha=0.35)
        self.assertEqual(len(samples), 100)
        self.assertTrue(all(len(v.number.alphas) == 1 and v.number.alpha == 0.35
                            for v in samples))
        numbers = [v.number for v in samples]
        self.assertEqual(len(np.unique(cluster_numbers(numbers, k=4))), 4)
        dist = pairwise_distances(numbers[:5])
        np.testing.assert_allclose(dist, dist.T, atol=1e-8)
        np.testing.assert_allclose(np.diag(dist), np.zeros(5), atol=1e-8)

    def test_threshold_extremes(self):
        numbers = [s.number for s in generate_samples(100)]
        self.assertEqual(len(set(cluster_numbers(numbers, threshold=1e-9))), 100)
        self.assertEqual(len(set(cluster_numbers(numbers, threshold=1e6))), 1)

    def test_generation_reproducible(self):
        s1 = generate_samples(100, seed=123, alpha=0.7)
        s2 = generate_samples(100, seed=123, alpha=0.7)
        self.assertEqual([(v.a, v.lam) for v in s1], [(v.a, v.lam) for v in s2])
        self.assertEqual([v.number.right[0] for v in s1],
                         [v.number.right[0] for v in s2])


if __name__ == '__main__':
    unittest.main()
