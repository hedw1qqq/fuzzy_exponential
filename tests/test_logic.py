import unittest
import numpy as np
from fuzzy_numbers import exponential, calculate, compare
from clustering import generate_samples, pairwise_distances, cluster_numbers


class FuzzyNumbersTest(unittest.TestCase):
    def test_alpha_cut(self):
        a = exponential(2, 0.5)
        self.assertAlmostEqual(a.left[-1], 2.0)
        self.assertAlmostEqual(a.right[-1], 2.0)
        self.assertAlmostEqual(a.right[0], 2-np.log(0.01)/0.5)

    def test_arithmetic(self):
        a, b = exponential(2, 1), exponential(3, 2)
        self.assertAlmostEqual(calculate(a,b,'+').left[-1], 5)
        self.assertAlmostEqual(calculate(a,b,'-').right[-1], -1)
        self.assertAlmostEqual(calculate(a,b,'*').left[-1], 6)
        self.assertAlmostEqual(calculate(a,b,'/').right[-1], 2/3)

    def test_compare(self):
        self.assertEqual(compare(exponential(1, 1), exponential(4, 1)), '<')

    def test_invalid_division(self):
        with self.assertRaisesRegex(ValueError, '0'):
            calculate(exponential(3,1), exponential(0,2), '/')

    def test_negative_intervals(self):
        a = exponential(-4,1)
        b = exponential(0.5,2)
        result = calculate(a,b,'*')
        self.assertTrue(np.all(result.left <= result.right))


class ClusteringTest(unittest.TestCase):
    def test_distance_at_large_shift(self):
        numbers = [exponential(1e8, 1), exponential(1e8 + 1, 1)]
        self.assertAlmostEqual(pairwise_distances(numbers)[0, 1], 1.0)

    def test_generation_and_k(self):
        samples = generate_samples(100)
        self.assertEqual(len(samples),100)
        numbers = [v.number for v in samples]
        self.assertEqual(len(np.unique(cluster_numbers(numbers,k=4))), 4)
        dist = pairwise_distances(numbers[:5])
        np.testing.assert_allclose(dist, dist.T, atol=1e-8)
        np.testing.assert_allclose(np.diag(dist), np.zeros(5), atol=1e-8)

    def test_threshold_extremes(self):
        numbers = [s.number for s in generate_samples(100)]
        labels = cluster_numbers(numbers, threshold=1e-9)
        self.assertEqual(len(set(labels)),100)
        labels2 = cluster_numbers(numbers, threshold=1e6)
        self.assertEqual(len(set(labels2)),1)

    def test_generation_reproducible(self):
        s1, s2 = generate_samples(100,seed=123), generate_samples(100,seed=123)
        self.assertEqual([v.a for v in s1], [v.a for v in s2])


if __name__ == '__main__':
    unittest.main()
