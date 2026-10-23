# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit seven shared MPM material parameters to the six recorded sand experiments.

Run with: uv run --with scikit-learn==1.7.2 python scripts/fit_sand.py --output RUN
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

from newton.examples.mpm.sand_material import DEFAULTS, FROZEN
from scripts.sand_sysid.cases import DEFAULT_CASES, load_cases
from scripts.sand_sysid.forest import BOUNDS, decode, propose, validate_bounds

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Replace checkpoints atomically so interrupted writes preserve the previous run."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def meets_target(trial, tolerance):
    """Require every selected experiment to meet the requested relative RMS tolerance."""
    return trial["valid"] and all(case["error"] <= tolerance for case in trial["scenarios"].values())


def load_start_point(path, *, bounds, cases, settings, suite_frames, frozen=None):
    """Load a valid saved fit as the first candidate of a new search."""
    frozen = FROZEN if frozen is None else frozen
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    previous_inputs = json.loads((path.parent / "inputs.json").read_text())
    if not result.get("valid") or not isinstance(result.get("material"), dict):
        raise ValueError(f"{path}: expected a valid fit_sand.py best.json or fit.json")
    if (
        previous_inputs.get("bounds") != bounds
        or previous_inputs.get("cases") != cases
        or previous_inputs.get("settings") != settings
        or previous_inputs.get("suite_frames") != suite_frames
    ):
        raise ValueError("Starting fit must use the same bounds, cases, scene settings, and frame count")
    point = result.get("point")
    material = result["material"]
    if set(material) != set(DEFAULTS) | set(FROZEN) or any(
        not math.isclose(material.get(name, float("nan")), value, rel_tol=1e-8, abs_tol=1e-10)
        for name, value in frozen.items()
    ):
        raise ValueError(f"{path}: missing or changed material parameters")
    decoded = decode(point, bounds)
    if any(not math.isclose(decoded[name], material[name], rel_tol=1e-8, abs_tol=1e-10) for name in bounds):
        raise ValueError(f"{path}: saved material does not match its optimizer point")
    return point, {"path": str(path), "evaluation": result.get("evaluation"), "error": result.get("error")}


def build_observe_command(result_path, *, cases=DEFAULT_CASES, device=None, experiment=None):
    """Recreate a fitted suite in the OpenGL viewer from a saved fit result."""
    from newton.examples.mpm.example_mpm_sand_tests import Example  # noqa: PLC0415

    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    inputs = json.loads((result_path.parent / "inputs.json").read_text())
    if not result.get("valid") or not isinstance(result.get("scenarios"), dict) or not result["scenarios"]:
        raise ValueError(f"{result_path}: expected a valid fit_sand.py best.json or fit.json result")
    material = result.get("material")
    if not isinstance(material, dict) or set(material) != set(DEFAULTS) | set(FROZEN):
        raise ValueError(f"{result_path}: expected all fitted and frozen MPM material parameters")
    if any(
        not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
        for value in material.values()
    ):
        raise ValueError(f"{result_path}: material parameters must be finite numbers")

    saved_cases = inputs.get("cases")
    if not isinstance(saved_cases, dict) or set(saved_cases) != set(result["scenarios"]):
        raise ValueError(f"{result_path}: case names do not match the neighboring inputs.json")
    catalog = load_cases(cases)
    if any(catalog.get(name) != saved_case for name, saved_case in saved_cases.items()):
        raise ValueError("Case data differs from the fit; pass the original manifest with --cases")
    if experiment is not None and experiment not in saved_cases:
        raise ValueError(f"Unknown experiment {experiment!r}; available: {', '.join(saved_cases)}")

    settings = inputs.get("settings")
    defaults = vars(Example.create_parser().parse_args([]))
    allowed_settings = {
        key
        for key, value in defaults.items()
        if isinstance(value, int | float) and not isinstance(value, bool) and key != "num_frames"
    } - (set(DEFAULTS) | set(FROZEN))
    if not isinstance(settings, dict) or set(settings) - allowed_settings:
        raise ValueError(f"{result_path}: inputs.json has unsupported suite settings")
    if any(
        not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
        for value in settings.values()
    ):
        raise ValueError(f"{result_path}: suite settings must be finite numbers")
    frames = inputs.get("suite_frames")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError(f"{result_path}: inputs.json has an invalid suite frame count")
    selected_device = device or inputs.get("device")
    if not isinstance(selected_device, str) or not selected_device:
        raise ValueError(f"{result_path}: inputs.json has no device")

    command = [
        sys.executable,
        "-m",
        "newton.examples",
        "mpm_sand_tests",
        "--viewer",
        "gl",
        "--device",
        selected_device,
        "--cases",
        str(Path(cases).resolve()),
        "--num-frames",
        str(frames),
    ]
    for name in [experiment] if experiment is not None else saved_cases:
        command.extend(("--case", name))
    for name, value in {**settings, **material}.items():
        command.extend(("--" + name.replace("_", "-"), str(value)))
    return command


def observe_sand(argv):
    parser = argparse.ArgumentParser(
        prog="fit_sand.py observe_sand", description="Open a saved sand fit in Newton's OpenGL viewer"
    )
    parser.add_argument("result", type=Path, help="best.json or fit.json written by fit_sand.py")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Original case manifest, if custom")
    parser.add_argument("--device", help="Override the device recorded in inputs.json")
    parser.add_argument("--experiment", help="Display only one named experiment from the saved fit")
    args = parser.parse_args(argv)
    try:
        command = build_observe_command(args.result, cases=args.cases, device=args.device, experiment=args.experiment)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "observe_sand":
        observe_sand(argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Open a saved result: fit_sand.py observe_sand PATH/TO/best.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--scenario", action="append", help="Optional subset; by default fit all six experiments")
    parser.add_argument("--bounds", type=Path, help="JSON search intervals; see scripts/sand_sysid/forest.py")
    parser.add_argument("--fixed-material", type=Path, help="JSON material values excluded from optimization")
    parser.add_argument(
        "--continuous", action="store_true", help="Keep evaluating until interrupted, ignoring budget and target"
    )
    parser.add_argument("--suite-settings", type=Path, help="Fixed numeric scene settings, never optimized")
    parser.add_argument(
        "--start-from", type=Path, help="Use a prior valid best.json or fit.json as the first candidate"
    )
    parser.add_argument("--suite-frames", type=int, default=720)
    parser.add_argument("--max-evaluations", type=int, default=64, help="Total budget, including resumed evaluations")
    parser.add_argument("--initial-samples", type=int, default=16)
    parser.add_argument(
        "--target-error", type=float, default=0.1, help="Stop when EVERY case has relative RMS error at most this value"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800, help="Maximum wall seconds per GPU evaluation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if min(args.max_evaluations, args.suite_frames) < 1 or args.initial_samples < 2 or args.seed < 0:
        parser.error("Use positive budgets, at least two initial samples, and a nonnegative seed")
    if not 0 <= args.target_error < float("inf") or not 0 < args.timeout < float("inf"):
        parser.error("Target error must be finite and nonnegative; timeout must be finite and positive")
    try:
        sklearn_version = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        parser.error("Run with: uv run --with scikit-learn==1.7.2 python scripts/fit_sand.py ...")
    bounds = json.loads(args.bounds.read_text()) if args.bounds else BOUNDS
    validate_bounds(bounds)
    bounds = {name: bounds[name] for name in BOUNDS}
    fixed = json.loads(args.fixed_material.read_text()) if args.fixed_material else {}
    if not isinstance(fixed, dict) or set(fixed) - set(BOUNDS):
        parser.error("Fixed material must be an object containing optimized material keys")
    frozen = {**FROZEN, **fixed}
    for name, value in fixed.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            parser.error(f"Invalid fixed value for {name}")
        if not bounds[name][0] <= value <= bounds[name][1]:
            parser.error(f"Fixed {name} is outside its bounds")
    bounds = {name: interval for name, interval in bounds.items() if name not in fixed}
    if not bounds:
        parser.error("At least one material parameter must remain free")
    cases = load_cases(args.case_manifest)
    if args.scenario:
        if set(args.scenario) - set(cases):
            parser.error("Unknown experiment name")
        cases = {name: case for name, case in cases.items() if name in args.scenario}
    settings = json.loads(args.suite_settings.read_text()) if args.suite_settings else {}
    if set(settings) & (set(DEFAULTS) | set(FROZEN)):
        parser.error("Material values belong to the optimizer, not --suite-settings")
    try:
        start_point, start_source = (
            load_start_point(
                args.start_from,
                bounds=bounds,
                cases=cases,
                settings=settings,
                suite_frames=args.suite_frames,
                frozen=frozen,
            )
            if args.start_from
            else (None, None)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    source_paths = [
        Path(__file__),
        *(ROOT / "scripts/sand_sysid").glob("*.py"),
        *(ROOT / "newton/examples/mpm").glob("*sand*.py"),
        ROOT / "newton/examples/mpm/example_mpm_bottle_drop.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(set(source_paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    config = {
        "cases": cases,
        "bounds": bounds,
        "frozen": frozen,
        "settings": settings,
        "suite_frames": args.suite_frames,
        "seed": args.seed,
        "initial_samples": args.initial_samples,
        "device": args.device,
        "start_from": start_source,
        "source_sha256": digest.hexdigest(),
        "sklearn_version": sklearn_version,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "warp-lang", "newton")},
        "optimizer": "random_forest_lcb",
        "loss": "mean-relative dimension RMS; weighted case RMS",
    }
    output = args.output.resolve()
    if args.resume:
        if json.loads((output / "inputs.json").read_text()) != config:
            parser.error("Resume inputs differ: keep material bounds, data, geometry, code, and seed unchanged")
        history = json.loads((output / "history.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "inputs.json", config)
        history = []
        write_json(output / "history.json", history)
    write_json(output / "scene-settings.json", settings)
    best = min((trial for trial in history if trial["valid"]), key=lambda trial: trial["error"], default=None)
    if best:
        write_json(output / "best.json", best)
    while args.continuous or (
        len(history) < args.max_evaluations and not any(meets_target(trial, args.target_error) for trial in history)
    ):
        point, strategy = (
            (start_point, "seeded_best")
            if not history and start_point is not None
            else propose(history, bounds, seed=args.seed, initial_samples=args.initial_samples)
        )
        material = {**decode(point, bounds), **frozen}
        directory = output / f"trial-{len(history) + 1:04d}"
        directory.mkdir(exist_ok=True)
        response = directory / "response.json"
        response.unlink(missing_ok=True)
        request = {
            "material": material,
            "scenarios": list(cases.values()),
            "device": args.device,
            "options": {
                "case_manifest": str(args.case_manifest.resolve()),
                "suite_settings": str(output / "scene-settings.json"),
                "suite_frames": args.suite_frames,
            },
        }
        write_json(directory / "request.json", request)
        print(f"Trial {len(history) + 1}/{args.max_evaluations} ({strategy}): {material}", flush=True)
        started = time.monotonic()
        with (directory / "simulation.log").open("w") as log:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.sand_sysid.evaluate",
                        str(directory / "request.json"),
                        str(response),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result = {"valid": False, "error": None, "scenarios": {}, "failure_reason": "Evaluation timed out"}
            else:
                if completed.returncode or not response.exists():
                    raise RuntimeError(
                        f"GPU worker failed; inspect {directory / 'simulation.log'}; resume after fixing the cause"
                    )
                result = json.loads(response.read_text())
        trial = {
            "evaluation": len(history) + 1,
            "point": point,
            "strategy": strategy,
            "material": material,
            "elapsed_seconds": time.monotonic() - started,
            **result,
        }
        history.append(trial)
        write_json(output / "history.json", history)
        if trial["valid"] and (best is None or trial["error"] < best["error"]):
            best = trial
            write_json(output / "best.json", best)
        print(
            f"  valid={trial['valid']}, combined error={trial['error']}, elapsed={trial['elapsed_seconds']:.1f}s",
            flush=True,
        )
        # A candidate meeting all per-case tolerances need not have the lowest aggregate loss.
        if meets_target(trial, args.target_error):
            write_json(output / "fit.json", trial)
            if not args.continuous:
                break
    fitting = next((trial for trial in history if meets_target(trial, args.target_error)), None)
    if fitting:
        write_json(output / "fit.json", fitting)
    else:
        (output / "fit.json").unlink(missing_ok=True)
    summary = {
        "evaluations": len(history),
        "target_error": args.target_error,
        "stop_reason": "target_met" if fitting else "evaluation_budget",
        "best_error": best["error"] if best else None,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if best is None:
        raise SystemExit("No valid physical candidate. Inspect trial logs and fixed scene settings before continuing.")


if __name__ == "__main__":
    main()
