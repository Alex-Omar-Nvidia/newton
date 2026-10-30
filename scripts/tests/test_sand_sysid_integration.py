# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded end-to-end fit using the recorded sand experiment CSVs and CUDA."""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from scripts.sand_sysid.cases import load_cases

ROOT = Path(__file__).resolve().parents[2]
POUR = "2m-gal-sand-pour"
BOTTLE = "1m-500ml-water-bottle-drop"


@unittest.skipUnless(wp.is_cuda_available(), "The sand fit integration test requires CUDA")
@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Run with --with scikit-learn==1.7.2")
class TestSandSysidIntegration(unittest.TestCase):
    """Exercise CSV loading, GPU evaluation, and optimizer checkpoints together."""

    def test_one_candidate_uses_recorded_pour_and_bottle_trials(self):
        """Fit one real candidate and verify the saved per-case and combined losses."""
        catalog = load_cases()
        self.assertEqual(len(catalog), 6)
        self.assertTrue(all(case["trial_count"] == 3 for case in catalog.values()))
        np.testing.assert_allclose(catalog[POUR]["target"], [0.6366666667, 0.52, 0.0966666667])
        np.testing.assert_allclose(catalog[BOTTLE]["target"], [0.1266666667, 0.13, 0.0333333333])

        # These settings retain the physical 8 s measurement window while
        # reducing particle count and frame count for a bounded CI run.
        settings = {
            "fps": 20,
            "substeps": 8,
            "pour_voxel_size": 0.025,
            "bottle_voxel_size": 0.02,
            "bed_width": 0.4,
            "bed_length": 0.4,
            "cap_radius": 0.02,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fit"
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(json.dumps(settings))
            command = [
                sys.executable,
                "scripts/fit_sand.py",
                "--output",
                str(output),
                "--scenario",
                POUR,
                "--scenario",
                BOTTLE,
                "--suite-settings",
                str(settings_path),
                "--suite-frames",
                "160",
                "--max-evaluations",
                "1",
                "--initial-samples",
                "2",
                "--target-error",
                "100",
                "--timeout",
                "50",
                "--device",
                "cuda:0",
            ]
            started = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=55, check=False)
            self.assertLess(time.monotonic() - started, 60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            inputs = json.loads((output / "inputs.json").read_text())
            history = json.loads((output / "history.json").read_text())
            best = json.loads((output / "best.json").read_text())
            fit = json.loads((output / "fit.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            response = json.loads((output / "trial-0001" / "response.json").read_text())
            self.assertEqual(list(inputs["cases"]), [POUR, BOTTLE])
            self.assertEqual(inputs["settings"], settings)
            self.assertEqual(inputs["suite_frames"], 160)
            for name in (POUR, BOTTLE):
                self.assertEqual(inputs["cases"][name]["trials"], catalog[name]["trials"])
                self.assertEqual(inputs["cases"][name]["target"], catalog[name]["target"])
                self.assertTrue(Path(inputs["cases"][name]["measurements_csv"]).is_file())

            self.assertEqual(len(history), 1)
            self.assertEqual(best, fit)
            self.assertEqual(best, history[0])
            self.assertTrue(best["valid"])
            self.assertEqual(best["strategy"], "baseline")
            self.assertEqual(best["scenarios"], response["scenarios"])
            self.assertEqual(set(best["scenarios"]), {POUR, BOTTLE})
            errors = []
            for name, case in best["scenarios"].items():
                self.assertTrue(case["valid"], (name, case))
                self.assertTrue(case["measurement"]["valid"])
                self.assertEqual(len(case["residuals"]), 3)
                self.assertTrue(math.isfinite(case["error"]))
                self.assertGreaterEqual(case["error"], 0)
                errors.append(case["error"])
            self.assertAlmostEqual(best["error"], math.sqrt(sum(error**2 for error in errors) / len(errors)))
            self.assertEqual(summary["evaluations"], 1)
            self.assertEqual(summary["stop_reason"], "target_met")
            self.assertEqual(summary["best_error"], best["error"])


if __name__ == "__main__":
    unittest.main()
