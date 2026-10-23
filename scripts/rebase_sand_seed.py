# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Represent a valid sand fit under new search bounds for a seeded next run."""

import argparse
import json
import math
from pathlib import Path

from newton.examples.mpm.sand_material import DEFAULTS, FROZEN
from scripts.sand_sysid.forest import decode, validate_bounds


def rebase_seed(source, bounds_path, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    result = json.loads(source.read_text())
    inputs = json.loads((source.parent / "inputs.json").read_text())
    bounds = json.loads(Path(bounds_path).read_text())
    validate_bounds(bounds)
    material = result.get("material")
    if not result.get("valid") or not isinstance(material, dict) or set(material) != set(DEFAULTS) | set(FROZEN):
        raise ValueError(f"{source}: expected a valid sand fit with every material parameter")
    if set(inputs.get("bounds", {})) != set(bounds):
        raise ValueError("Source and new bounds must contain the same optimized parameters")

    point = []
    for name, (lo, hi, scale, offset) in bounds.items():
        value = material[name]
        if not isinstance(value, int | float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{name}: saved value {value!r} is outside the new bounds")
        coordinate = (
            math.log((value + offset) / (lo + offset)) / math.log((hi + offset) / (lo + offset))
            if scale == "log"
            else (value - lo) / (hi - lo)
        )
        point.append(coordinate)
    decoded = decode(point, bounds)
    if any(not math.isclose(decoded[name], material[name], rel_tol=1e-8, abs_tol=1e-10) for name in bounds):
        raise ValueError("Rebased point does not reproduce the saved material")

    rebased_result = {**result, "point": point, "rebased_from": str(source)}
    rebased_inputs = {**inputs, "bounds": bounds, "rebased_seed_from": str(source)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "best.json").write_text(json.dumps(rebased_result, indent=2, allow_nan=False) + "\n")
    (output / "inputs.json").write_text(json.dumps(rebased_inputs, indent=2, allow_nan=False) + "\n")
    return output / "best.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Valid best.json or fit.json from a completed run")
    parser.add_argument("--bounds", type=Path, required=True, help="New search bounds JSON")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the rebased seed")
    args = parser.parse_args()
    print(rebase_seed(args.source, args.bounds, args.output))
