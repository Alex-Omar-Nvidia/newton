# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from scripts.sand_sysid.cases import dimension_error, load_cases


class TestSandCases(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
