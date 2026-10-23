# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

from newton.examples.mpm.example_mpm_bottle_drop import Example as BottleExample
from newton.examples.mpm.example_mpm_bottle_drop import bottle_mesh, measure_crater, surface_grid
from scripts.sand_sysid.cases import dimension_error, load_cases


class TestSandCases(unittest.TestCase):
    def test_reject_shallow_bottle_bed(self):
        """Reject bottle tests with less than 30 cm of sand before allocating physics."""
        args = BottleExample.create_parser().parse_args([])
        args.bed_depth = 0.29
        with self.assertRaisesRegex(ValueError, "0.30 m"):
            BottleExample(None, args)

    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_bottle_configurations_and_measured_targets(self):
        """Load all four bottle datasets with water masses and crater mean targets."""
        expected = {
            "1m-1l-water-bottle-drop": (1.0, 1.0, [0.16, 0.16, 0.1 / 3]),
            "1m-500ml-water-bottle-drop": (1.0, 0.5, [0.38 / 3, 0.13, 0.1 / 3]),
            "2m-1l-water-bottle-drop": (2.0, 1.0, [0.52 / 3, 0.16, 0.13 / 3]),
            "2m-500ml-water-bottle-drop": (2.0, 0.5, [0.47 / 3, 0.47 / 3, 0.04]),
        }
        bottles = {name: case for name, case in self.cases.items() if case["test_type"] == "bottle_drop"}
        self.assertEqual(set(bottles), set(expected))
        for name, (drop_height, mass, means) in expected.items():
            with self.subTest(configuration=name):
                case = bottles[name]
                self.assertEqual(case["trial_count"], 3)
                self.assertEqual(case["dimensions"], ["width_m", "length_m", "depth_m"])
                config = case["configuration"]
                self.assertEqual(config["drop_height_m"], drop_height)
                self.assertEqual(config["total_mass_kg"], mass)
                self.assertEqual(config["orientation"], "cap_down")
                self.assertTrue(config["bottle_removed_before_measurement"])
                self.assertIsNone(config["cap_diameter_m"])
                for actual, expected_mean in zip(case["target"], means, strict=True):
                    self.assertAlmostEqual(actual, expected_mean)

    def test_crater_depth_loss_and_zero_variance(self):
        """Score crater depth correctly even when repeated trials have zero variance."""
        case = self.cases["2m-500ml-water-bottle-drop"]
        self.assertEqual(case["sample_std_m"][2], 0.0)
        measurement = dict(zip(case["dimensions"], case["target"], strict=True))
        self.assertEqual(dimension_error(measurement, case)[1], 0.0)
        measurement["depth_m"] *= 1.1
        residuals, magnitude = dimension_error(measurement, case)
        self.assertAlmostEqual(residuals[2], 0.1)
        self.assertAlmostEqual(magnitude, 0.1 / math.sqrt(3))
        measurement["height_m"] = measurement.pop("depth_m")
        with self.assertRaises(KeyError):
            dimension_error(measurement, case)

    def test_pour_cases_retain_volume_and_trials(self):
        """Keep the renamed two-meter dataset tied to five gallons and pile height."""
        self.assertEqual(len(self.cases), 6)
        case = self.cases["2m-gal-sand-pour"]
        self.assertEqual(case["configuration"]["sand_volume_us_gallons"], 5.0)
        self.assertEqual(case["dimensions"][2], "height_m")
        self.assertEqual(case["trials"], [[0.59, 0.6, 0.1], [0.72, 0.53, 0.08], [0.6, 0.43, 0.11]])

    def test_crater_measurement(self):
        """Measure a known central excavation without counting a detached depression."""
        reference = np.full((40, 40), 0.12)
        current = reference.copy()
        current[15:25, 16:24] -= 0.04
        current[2:5, 2:5] -= 0.03
        metrics = measure_crater(reference, current, cell_size=0.01, threshold=0.02)
        np.testing.assert_allclose([metrics["width_m"], metrics["length_m"], metrics["depth_m"]], [0.10, 0.08, 0.04])
        self.assertFalse(metrics["touches_measurement_boundary"])
        self.assertEqual(measure_crater(reference, reference, cell_size=0.01, threshold=0.002)["depth_m"], 0)
        points = np.array([[-0.1, -0.1, 0.02], [-0.1, -0.1, 0.08], [0.1, 0.1, 0.04]])
        surface, spacing = surface_grid(points, np.full(3, 0.005), extent=0.2, cell_size=0.2)
        np.testing.assert_allclose(surface, [[0.085, 0], [0, 0.045]])
        self.assertEqual(spacing, 0.2)
        # A raised rim must increase depth even when the original surface is unchanged.
        raised = reference.copy()
        raised[12:28, 13:27] = 0.15
        raised[15:25, 16:24] = 0.08
        rim_metrics = measure_crater(reference, raised, cell_size=0.01, threshold=0.02)
        self.assertAlmostEqual(rim_metrics["depth_m"], 0.07)
        shifted_reference = reference + 0.005
        self.assertAlmostEqual(
            measure_crater(shifted_reference, raised, cell_size=0.01, threshold=0.02)["depth_m"], 0.07
        )

    def test_bottle_geometry(self):
        """Verify a closed outward-facing bottle collider and positive mass properties."""
        vertices, indices, com_z, inertia = bottle_mesh(0.04, 0.25, 0.015, 0.015)
        triangles = indices.reshape(-1, 3)
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2))
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        self.assertGreater(float(np.sum(a * np.cross(b, c)) / 6), 0)
        self.assertTrue(0 < com_z < 0.25)
        self.assertTrue(all(value > 0 for value in inertia))


if __name__ == "__main__":
    unittest.main()
