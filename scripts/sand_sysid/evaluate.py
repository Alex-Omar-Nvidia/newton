# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated GPU worker for one material candidate; invoked by fit_sand.py."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.mpm.example_mpm_sand_tests import Example as SuiteExample
from scripts.sand_sysid.cases import dimension_error


def run_suite(material, options, scenarios):
    """Evaluate one material candidate against the simultaneous experiment suite."""
    args = SuiteExample.create_parser().parse_args([])
    if options.suite_settings:
        settings = json.loads(options.suite_settings.read_text())
        allowed = {key for key, value in vars(args).items() if isinstance(value, int | float) and key != "num_frames"}
        if set(settings) - allowed:
            raise ValueError(f"Unknown or nonphysical suite settings: {set(settings) - allowed}")
        for key, value in settings.items():
            setattr(args, key, value)
    args.cases = options.case_manifest
    args.case = [scenario["name"] for scenario in scenarios]
    args.viewer = "null"
    args.num_frames = options.suite_frames
    for key, value in material.items():
        setattr(args, key, value)
    viewer = newton.viewer.ViewerNull()
    suite = SuiteExample(viewer, args)
    try:
        for _ in range(args.num_frames):
            suite.step()
        results = suite.measure()
        for case, child in zip(suite.cases, suite.children, strict=True):
            try:
                child.test_final()
            except AssertionError as exc:
                results[case["name"]]["measurement"].update(valid=False, failure_reason=str(exc))
        return {name: value["measurement"] for name, value in results.items()}, suite.resolved
    finally:
        viewer.close()


def aggregate_error(errors, weights):
    """Combine scenario RMS errors with explicit scenario weights."""
    return float(np.sqrt(np.average(np.square(errors), weights=weights)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    options = SimpleNamespace(**request["options"])
    options.case_manifest = Path(options.case_manifest)
    options.suite_settings = Path(options.suite_settings) if options.suite_settings else None
    wp.init()
    wp.set_device(request["device"])
    result = {"valid": False, "error": None, "scenarios": {}}
    try:
        measurements, resolved = run_suite(request["material"], options, request["scenarios"])
        result["resolved_simulation_settings"] = resolved
        errors, weights = [], []
        for case in request["scenarios"]:
            measurement = measurements[case["name"]]
            entry = {"measurement": measurement, "valid": measurement["valid"]}
            if entry["valid"]:
                residuals, error = dimension_error(measurement, case)
                entry.update(residuals=residuals, error=error)
                errors.append(error)
                weights.append(case["weight"])
            result["scenarios"][case["name"]] = entry
        result["valid"] = len(errors) == len(request["scenarios"])
        if result["valid"]:
            result["error"] = aggregate_error(errors, weights)
    except (ValueError, RuntimeError, AssertionError) as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    args.response.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
