# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load experimental configurations and derive targets from unmodified trial CSVs.

Inspect the registered cases with ``uv run -m scripts.sand_sysid.cases``.
Run all cases with ``uv run -m newton.examples mpm_sand_tests`` or fit shared
material parameters with ``uv run --with scikit-learn==1.7.2 python scripts/fit_sand.py --output RUN``.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

DEFAULT_CASES = Path(__file__).with_name("cases.json")


def load_cases(path=DEFAULT_CASES):
    """Load measured trials, mean targets, and shared configuration per test type."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    result = {}
    for case in manifest["cases"]:
        name = case["name"]
        if not name or name in result:
            raise ValueError(f"Case names must be nonempty and unique: {name!r}")
        definition = manifest["test_types"][case["test_type"]]
        dimensions = definition["dimensions"]
        csv_path = path.parent / case["measurements_csv"]
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != dimensions:
                raise ValueError(f"{csv_path}: expected columns {dimensions}, got {reader.fieldnames}")
            try:
                trials = [[float(row[dimension]) for dimension in dimensions] for row in reader]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{csv_path}: missing or nonnumeric measurement") from exc
        if not trials or any(not math.isfinite(value) or value <= 0 for trial in trials for value in trial):
            raise ValueError(f"{csv_path}: measurements must be finite and positive")
        mean = [statistics.mean(column) for column in zip(*trials, strict=True)]
        spread = [statistics.stdev(column) if len(trials) > 1 else None for column in zip(*trials, strict=True)]
        configuration = {**definition["shared"], **case["configuration"]}
        if case["test_type"] == "bottle_drop":
            configuration["total_mass_kg"] = (
                configuration["water_volume_ml"] * 1.0e-6 * configuration["water_density_kg_m3"]
                + configuration["empty_bottle_mass_kg"]
            )
        result[name] = {
            "name": name,
            "test_type": case["test_type"],
            "configuration": configuration,
            "unspecified_configuration": [key for key, value in configuration.items() if value is None],
            "measurements_csv": str(csv_path.resolve()),
            "dimensions": dimensions,
            "trials": trials,
            "trial_count": len(trials),
            "target": mean,
            "sample_std_m": spread,
            # Relative residuals remain defined when repeated measurements have zero variance.
            "scales": mean.copy(),
            "weight": 1.0,
        }
    return result


def dimension_error(measurement, target):
    """Return signed relative residuals and RMS error against a case's mean target.

    The third dimension is explicitly height for pours or positive crater depth for
    bottle drops. Standard deviation is retained as descriptive trial spread; it is
    not automatically treated as measurement uncertainty or used as a divisor.
    """
    dimensions = target.get("dimensions", ("width_m", "length_m", "height_m"))
    if len(dimensions) != len(target["target"]) or len(dimensions) != len(target["scales"]):
        raise ValueError("Dimensions, targets and scales must have matching lengths")
    if not dimensions:
        raise ValueError("At least one dimension is required")
    residuals = []
    for dimension, expected, scale in zip(dimensions, target["target"], target["scales"], strict=True):
        value = float(measurement[dimension])
        if not all(math.isfinite(x) for x in (value, expected, scale)) or value < 0 or expected <= 0 or scale <= 0:
            raise ValueError("Measurements must be finite and nonnegative; targets and scales must be positive")
        residuals.append((value - expected) / scale)
    return residuals, math.sqrt(statistics.mean(value**2 for value in residuals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--test-type", choices=("sand_pour", "bottle_drop"))
    args = parser.parse_args()
    for case in load_cases(args.cases).values():
        if args.test_type and case["test_type"] != args.test_type:
            continue
        print(json.dumps(case, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
