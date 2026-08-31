import unittest
import numpy as np
from mpcrl.planning_residual import expand_temporal_residual


class T(unittest.TestCase):
    def test_shape_and_first_action(self):
        r = np.array([1.0, -2.0], dtype=np.float32)
        x = expand_temporal_residual(r, 3, 2, decay=0.5)
        self.assertEqual(x.shape, (3, 2))
        np.testing.assert_allclose(x[0], r)
        np.testing.assert_allclose(x[1], 0.5 * r)
        np.testing.assert_allclose(x[2], 0.25 * r)

    def test_decay_is_bounded(self):
        x = expand_temporal_residual([1.0], 4, 1, decay=2.0)
        np.testing.assert_allclose(x[:, 0], np.ones(4))


if __name__ == '__main__':
    unittest.main()
