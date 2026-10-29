# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.mpm.example_mpm_sand_tests import Example
from newton.examples.mpm.sand_material import DEFAULTS, FROZEN, apply_material
from newton.solvers import SolverImplicitMPM
from scripts.fit_sand import build_observe_command, load_start_point, meets_target, write_json
from scripts.sand_sysid.cases import DEFAULT_CASES, load_cases
from scripts.sand_sysid.forest import BOUNDS, decode, initial_point, propose, validate_bounds


class TestSandForest(unittest.TestCase):
    def test_material_assignment(self):
        """Apply all seven nondefault values and registered frozen defaults to a real model."""
        values = dict(zip(DEFAULTS, [2e6, 0.2, 0.01, 0.8, 5e5, 0.1, 250], strict=True))
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            SolverImplicitMPM.register_custom_attributes(builder)
            builder.add_particle(pos=wp.vec3(0, 0, 1), vel=wp.vec3(0), mass=1, radius=0.01)
            model = builder.finalize()
            apply_material(model, SimpleNamespace(**values))
            for name, value in {**values, **FROZEN}.items():
                np.testing.assert_allclose(getattr(model.mpm, name).numpy(), value, rtol=1e-6)

    def test_parameter_domain(self):
        """Round-trip the baseline and preserve physical limits including zero damping and yield."""
        validate_bounds(BOUNDS)
        baseline = decode(initial_point(BOUNDS), BOUNDS)
        for name, value in DEFAULTS.items():
            self.assertAlmostEqual(baseline[name], value, delta=max(1e-8, value * 1e-12))
        self.assertEqual(set(baseline), set(DEFAULTS))
        self.assertFalse(set(baseline) & set(FROZEN))
        with self.assertRaises(ValueError):
            validate_bounds({**BOUNDS, "poisson_ratio": [0.1, 0.5, "linear", 0]})
        with self.assertRaises(ValueError):
            decode([np.nan] * 7, BOUNDS)
        fixed = {"young_modulus": 9576000.0, "poisson_ratio": 0.25}
        free_bounds = {name: value for name, value in BOUNDS.items() if name not in fixed}
        history = []
        for _ in range(18):
            point, _ = propose(history, free_bounds, pool_size=32)
            material = {**decode(point, free_bounds), **FROZEN, **fixed}
            self.assertEqual(len(point), 5)
            self.assertEqual(material["young_modulus"], 9576000.0)
            self.assertEqual(material["poisson_ratio"], 0.25)
            history.append({"point": point, "valid": True, "error": sum(point)})

    def test_forest_search_and_replay(self):
        """Improve a bounded numerical objective with an actual forest and replay proposals from JSON."""
        bounds = {"x": [0, 1, "linear", 0], "y": [0, 1, "linear", 0]}
        history = [{"point": [0.9, 0.9], "valid": True, "error": 0.98}]
        strategies = []
        for _ in range(30):
            point, strategy = propose(history, bounds, seed=7, initial_samples=5, pool_size=512)
            self.assertTrue(np.all((np.array(point) >= 0) & (np.array(point) <= 1)))
            self.assertNotIn(point, [entry["point"] for entry in history])
            error = float(np.sum((np.array(point) - [0.2, 0.2]) ** 2))
            history.append({"point": point, "valid": True, "error": error})
            strategies.append(strategy)
        self.assertIn("random_forest", strategies)
        self.assertLess(min(entry["error"] for entry in history), 0.03)
        history.append({"point": [1, 1], "valid": False, "error": None})
        self.assertEqual(
            propose(history, bounds, seed=7, initial_samples=5, pool_size=512),
            propose(json.loads(json.dumps(history)), bounds, seed=7, initial_samples=5, pool_size=512),
        )

    def test_checkpoint_and_per_case_stopping(self):
        """Preserve checkpoints and prevent good aggregate losses from masking a failed case."""
        trial = {"valid": True, "error": 0.05, "scenarios": {"pour": {"error": 0.01}, "bottle": {"error": 0.2}}}
        self.assertFalse(meets_target(trial, 0.1))
        self.assertTrue(meets_target(trial, 0.25))
        self.assertFalse(meets_target({"valid": False}, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            write_json(path, [trial])
            self.assertEqual(json.loads(path.read_text()), [trial])
            with self.assertRaises(ValueError):
                write_json(path, [float("nan")])
            self.assertEqual(json.loads(path.read_text()), [trial])

    def test_observe_saved_fit_uses_recorded_simulation_settings(self):
        """Launch arguments must replay the actual fitted material and scene."""
        cases = load_cases()
        selected = list(cases)[-1]
        material = {**DEFAULTS, **FROZEN}
        material["friction"] = 0.81
        material["yield_stress"] = 250.0
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "best.json"
            write_json(
                result,
                {"valid": True, "material": material, "scenarios": {name: {"error": 0.08} for name in cases}},
            )
            write_json(
                result.parent / "inputs.json",
                {
                    "cases": cases,
                    "settings": {"pour_voxel_size": 0.025, "ground_friction": 0.7},
                    "suite_frames": 480,
                    "device": "cuda:0",
                },
            )
            command = build_observe_command(result, cases=DEFAULT_CASES)
            self.assertEqual(command[1:4], ["-m", "newton.examples", "mpm_sand_tests"])
            args = Example.create_parser().parse_args(command[4:])
            self.assertEqual(args.viewer, "gl")
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(args.case, list(cases))
            self.assertEqual(args.num_frames, 480)
            self.assertEqual(args.pour_voxel_size, 0.025)
            self.assertEqual(args.ground_friction, 0.7)
            self.assertEqual(args.friction, 0.81)
            self.assertEqual(args.yield_stress, 250.0)
            self.assertEqual(args.poisson_ratio, material["poisson_ratio"])
            self.assertEqual(args.hardening_rate, material["hardening_rate"])
            selected_command = build_observe_command(result, experiment=selected)
            selected_args = Example.create_parser().parse_args(selected_command[4:])
            self.assertEqual(selected_args.case, [selected])
            with self.assertRaisesRegex(ValueError, "Unknown experiment"):
                build_observe_command(result, experiment="not-a-case")
            write_json(result, {"valid": False, "material": material, "scenarios": {name: {} for name in cases}})
            with self.assertRaisesRegex(ValueError, "valid fit_sand.py"):
                build_observe_command(result)

    def test_new_search_starts_from_saved_best_point(self):
        """Use the saved best material as the first candidate only for matching conditions."""
        cases = load_cases()
        point = initial_point(BOUNDS)
        material = {**decode(point, BOUNDS), **FROZEN}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            write_json(path, {"valid": True, "point": point, "material": material, "evaluation": 42, "error": 0.16})
            write_json(
                path.parent / "inputs.json",
                {"bounds": BOUNDS, "cases": cases, "settings": {}, "suite_frames": 720},
            )
            loaded, source = load_start_point(path, bounds=BOUNDS, cases=cases, settings={}, suite_frames=720)
            self.assertEqual(loaded, point)
            self.assertEqual(source["evaluation"], 42)
            with self.assertRaisesRegex(ValueError, "same bounds, cases"):
                load_start_point(path, bounds=BOUNDS, cases=cases, settings={"fps": 30}, suite_frames=720)


if __name__ == "__main__":
    unittest.main()
