import unittest

import numpy as np

from clustering import alpha_mask, generate_samples, k_medoids, pairwise_distances
from fuzzy_numbers import FuzzyNumber, calculate, compare, exponential_membership


class FuzzyNumbersTest(unittest.TestCase):
    def test_membership_and_filter(self):
        intervals = [FuzzyNumber(0, 0), FuzzyNumber(0, 2), FuzzyNumber(1, 3)]
        np.testing.assert_allclose(exponential_membership([-1, 0, 1], 2), [0, 1, np.exp(-2)])
        np.testing.assert_array_equal(alpha_mask(intervals, 1, np.exp(-1)), [True, True, False])
        np.testing.assert_array_equal(alpha_mask(intervals, 1, 0), [True, True, True])
        np.testing.assert_array_equal(alpha_mask(intervals, 1, 1), [True, False, False])
        with self.assertRaises(ValueError):
            alpha_mask(intervals, 1, 1.1)

    def test_arithmetic(self):
        a, b = FuzzyNumber(2, 4), FuzzyNumber(1, 2)
        for op, expected in [('+', (3, 6)), ('-', (0, 3)), ('*', (2, 8)), ('/', (1, 4))]:
            with self.subTest(operation=op):
                self.assertEqual(calculate(a, b, op), FuzzyNumber(*expected))
        self.assertEqual(calculate(FuzzyNumber(-4, 2), FuzzyNumber(-2, -1), '/'), FuzzyNumber(-2, 4))
        with self.assertRaisesRegex(ValueError, '0'):
            calculate(a, FuzzyNumber(-1, 0), '/')

    def test_compare(self):
        self.assertEqual(compare(FuzzyNumber(1, 2), FuzzyNumber(3, 4)), '<')
        self.assertEqual(compare(FuzzyNumber(2, 3), FuzzyNumber(0, 2)), '≈')
        self.assertEqual(compare(FuzzyNumber(2, 3), FuzzyNumber(-3, -1)), '>')


class ClusteringTest(unittest.TestCase):
    def test_four_distances(self):
        numbers = [FuzzyNumber(0, 3), FuzzyNumber(1, 6)]
        for metric, distance in enumerate((3, 2, 16, np.sqrt(10))):
            with self.subTest(metric=metric):
                np.testing.assert_allclose(pairwise_distances(numbers, metric), [[0, distance], [distance, 0]])

    def test_k_medoids(self):
        numbers = [FuzzyNumber(x, x + 1) for x in (0, 1, 2, 10, 11, 12)]
        distances = pairwise_distances(numbers)
        labels, medoids = k_medoids(distances, k=2)
        self.assertEqual(set(medoids), {1, 4})
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])

    def test_generation(self):
        numbers = generate_samples(300, lam=2, seed=123)
        self.assertEqual(numbers, generate_samples(300, lam=2, seed=123))
        self.assertEqual(len(numbers), 300)
        self.assertTrue(all(0 <= item.left < item.right for item in numbers))
        self.assertGreater(sum(alpha_mask(numbers, 2, 0.8)), 0)


if __name__ == '__main__':
    unittest.main()
