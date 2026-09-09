import unittest

import numpy as np

from mpcrl.lewm_rl import (
    apply_action_residual,
    effective_rank,
    normalize_action,
    residual_ramp,
)


class TestLeWMRLUtilities(unittest.TestCase):
    def test_normalize_action(self):
        low = np.array([-2.0, -1.0], dtype=np.float32)
        high = np.array([2.0, 3.0], dtype=np.float32)
        mid = (low + high) * 0.5
        got = normalize_action(mid, low, high)
        np.testing.assert_allclose(got, np.zeros_like(got), atol=1e-6)

    def test_action_residual_is_bounded(self):
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        base = np.array([0.95, -0.95], dtype=np.float32)
        residual = np.array([1.0, -1.0], dtype=np.float32)
        action, effective = apply_action_residual(
            base, residual, low, high, scale=0.2, ramp=1.0
        )
        self.assertTrue(np.all(action <= high + 1e-6))
        self.assertTrue(np.all(action >= low - 1e-6))
        np.testing.assert_allclose(effective, [0.2, -0.2], atol=1e-6)

    def test_residual_ramp(self):
        self.assertEqual(residual_ramp(10, 20, 100), 0.0)
        self.assertAlmostEqual(residual_ramp(70, 20, 100), 0.5)
        self.assertEqual(residual_ramp(200, 20, 100), 1.0)

    def test_effective_rank(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2048, 8))
        r = effective_rank(x)
        self.assertGreater(r, 6.0)
        self.assertLessEqual(r, 8.0 + 1e-6)


if __name__ == '__main__':
    unittest.main()
