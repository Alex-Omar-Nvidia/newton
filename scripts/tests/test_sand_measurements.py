# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from newton.examples.mpm.example_mpm_sand_bucket import measure_pile
from scripts.sand_sysid.cases import dimension_error
from scripts.sand_sysid.evaluate import aggregate_error


class TestSandMeasurements(unittest.TestCase):
    def test_measurement_axes_and_retained_sand(self):
        """Measure known extents and expose material remaining above the outlet."""
        q = np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 0.4]])
        radii = np.full(2, 0.01)
        m = measure_pile(q, np.zeros_like(q), radii, outlet_height=1.0)
        np.testing.assert_allclose([m["width_m"], m["length_m"], m["height_m"]], [2.02, 4.02, 0.41])
        self.assertEqual(m["retained_fraction"], 0.0)
        rotated = measure_pile(q, np.zeros_like(q), radii, outlet_height=0.3, yaw_degrees=90)
        np.testing.assert_allclose([rotated["width_m"], rotated["length_m"]], [4.02, 2.02])
        self.assertEqual(rotated["retained_fraction"], 0.5)
        with self.assertRaises(ValueError):
            measure_pile(q * np.nan, np.zeros_like(q), radii, outlet_height=1.0)

    def test_normalized_errors(self):
        """Weight dimensional residuals without allowing their signs to cancel."""
        target = {"target": [2, 4, 0.5], "scales": [2, 4, 0.5]}
        residual, error = dimension_error({"width_m": 2.2, "length_m": 3.6, "height_m": 0.5}, target)
        np.testing.assert_allclose(residual, [0.1, -0.1, 0])
        self.assertAlmostEqual(error, np.sqrt(0.02 / 3))
        self.assertAlmostEqual(aggregate_error([error, 0], [1, 3]), error / 2)
        self.assertEqual(
            dimension_error(dict(zip(("width_m", "length_m", "height_m"), target["target"], strict=True)), target)[1], 0
        )


if __name__ == "__main__":
    unittest.main()
