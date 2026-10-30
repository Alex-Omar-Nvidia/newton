# Sand pile system identification

## Recorded experiments

`cases.json` registers two sand pours and one bottle-drop test type with four
configurations (1 m or 2 m, with 1 L or 500 mL of water). Each configuration points
to its own three-trial CSV in `data/`. The bottle trials measure crater width,
length, and positive depth after removing the bottle from a flattened mound of
completely dry sand. The bottles fall cap-first. Their masses are derived as 1 kg
and 0.5 kg from their full water volumes, neglecting the empty bottles as instructed.

Inspect the configurations, targets, individual trials, and sample standard deviations:

```sh
uv run -m scripts.sand_sysid.cases
uv run -m scripts.sand_sysid.cases --test-type bottle_drop
```

The target for each dimension is the mean of its three trials. The error function
uses relative RMS error against that mean, with depth replacing height for bottle
cases. Trial standard deviations are descriptive, not automatic error weights;
several observed dimensions have zero sample variance. Raw observations remain intact.

## Six-case simulation and calibration

```sh
uv run -m newton.examples mpm_sand_tests --results /tmp/sand-tests.json
uv run --with scikit-learn==1.7.2 python scripts/fit_sand.py --max-evaluations 64 --output /tmp/sand-fit
```

All six cases advance on one clock and appear in one scene, with 2 m clear gaps
between test tiles. Each uses its own MPM state and solver to prevent interference.
CUDA graph capture is enabled automatically on CUDA devices. The persistent
sparse grids are checked for overflow after replay. Each bottle phase has its
own capture so release and extraction retain their correct contact behavior.
Use `--no-cuda-graph` to compare eager stepping at the same resolution; CPU
devices use eager stepping automatically. Initial compilation and capture still
incur startup cost, and graph capture does not remove the GPU physics workload.

Rows contain the two pours, the 1 m bottle drops, and the 2 m bottle drops.
Bottle columns contain 1 L and 500 mL. Use the viewer's case controls to focus a
test, or repeat `--case NAME` to select cases. The calibration command selects
cases with repeated `--scenario NAME` arguments and shares material parameters.

Bottles settle above the bed for 0.5 s, fall cap-first, and begin vertical removal
3 s after release. Extraction travels 0.6 m at 0.25 m/s. Craters are measured
after removal and settling. Water contributes rigid mass and inertia; sloshing
and bottle deformation are not modeled.

**Crater depth is the post-removal rim elevation minus the crater floor
elevation**, measured on a smoothed surface heightfield. The original surface is
used only to locate the connected depression and measure its X/Y footprint.
The rim is the highest surface in a three-cell band around that depression;
the floor is its lowest surface. A crater reaching the measurement boundary is
invalid. Width, length, and this rim-to-floor depth feed the relative RMS error.
The overall error is the RMS of the six case errors and is unavailable until
every case has a valid settled observation.

Unknown experimental geometry remains `null` in the catalog. Runnable defaults
are provisional: a 0.45 x 0.45 m sand bed, 0.30 m deep, retained by four static
walls extending 0.05 m above the fill. `--bed-depth` accepts deeper beds and
rejects values below 0.30 m. The walls use the ground contact friction.
Bottle radii/heights are 0.04/0.25 m (1 L) and 0.033/0.20 m (500 mL);
cap radius/height 0.015/0.015 m. CLI options override these defaults. Calibration
accepts the same numeric settings as a JSON object via `--suite-settings`.
Results record the resolved geometry. Confirm these inputs before interpreting
a fitted material as identified from the experiments.

Run the bottle implementation independently:

```sh
uv run -m newton.examples mpm_bottle_drop
```

## Bounded integration check

Run the bounded end-to-end GPU check with the repository CSV catalog:

```sh
uv run --with scikit-learn==1.7.2 python -m unittest scripts.tests.test_sand_sysid_integration
```

The test loads all six three-trial CSVs, then fits one candidate against one
pour and one bottle drop. It uses an 8 s physical window at 20 FPS and coarser
test geometry. It verifies the saved case inputs, valid measurements, per-case
errors, best result, and summary. The child process has a 55 s deadline, so
the integration check remains below one minute. Full-resolution calibration
continues to use the settings described below.

## Random-forest material fitting

`fit_sand.py` fits **one shared material** to all selected experiments. Every
candidate runs all six cases headlessly with CUDA graphs. The initial design
contains the baseline, individual parameter probes, and space-filling samples.
After 16 initial evaluations, a 128-tree random forest models the log-transformed
combined error and proposes candidates with a lower-confidence-bound heuristic.
Tree disagreement encourages exploration; it is not calibrated uncertainty.
Twenty percent of later proposals continue space-filling exploration.

The search uses these exploratory bounds (all values are SI):

| Optimized attribute | Bounds | Coordinates |
| --- | --- | --- |
| `mpm:young_modulus` | 100,000–100,000,000 Pa | logarithmic |
| `mpm:poisson_ratio` | 0.05–0.45 | linear |
| `mpm:damping` | 0–0.1 s | logarithmic with 0.0001 s offset |
| `mpm:friction` | 0.1–1.5 | linear |
| `mpm:yield_pressure` | 1,000–1,000,000,000,000 Pa | logarithmic |
| `mpm:tensile_yield_ratio` | 0–1 | linear |
| `mpm:yield_stress` | 0–100,000 Pa | logarithmic with 100 Pa offset |

`yield_stress` is optimized automatically by `fit_sand.py`, starting from a
zero-stress baseline. No additional flag is needed.

Plasticity controls stay fixed at Newton's registered defaults throughout fitting:
`hardening=0`, `hardening_rate=1`, `softening_rate=1`, `dilatancy=0`, `viscosity=0`.
All twelve fields are assigned before constructing each solver. The material
cannot be overridden through the geometry/settings file. `--bounds FILE` accepts
exactly the seven optimized keys, each mapping to `[lower, upper, "linear" or
"log", offset]`; see `forest.py` for the default JSON-compatible structure.

Targets are the means of the three recorded trials. Per-case loss is the RMS of
three relative dimensional residuals. The shared loss is the weighted RMS of
case losses (equal weights by default). The run stops when **every** case has
relative RMS error at most `--target-error 0.1`, or the evaluation budget is spent.
Ten percent RMS does not imply every individual dimension is within ten percent.
Budget exhaustion is not convergence. Seven properties may not be uniquely
identifiable from these final dimensions; geometry and contact assumptions also
influence the fit. A finer-resolution validation remains necessary.

Each candidate runs in a separate process to release its GPU allocations and
isolate failures. An unsettled or invalid candidate cannot become the best fit;
its forest training loss is worse than all valid observations. Worker startup
failures stop the run with a log path rather than silently labeling the material
bad. Each candidate has a configurable `--timeout` (default 1,800 seconds).

The output directory contains:

- `inputs.json`: experimental trials, fixed settings, bounds, frozen values,
  seed, runtime versions, and workflow source fingerprint.
- `history.json`: atomic checkpoint of all completed trials and per-case results.
- `best.json`: lowest combined error from an actual valid simulation.
- `fit.json`: a candidate meeting every case's stopping tolerance, if found.
- `summary.json`: whether the tolerance or budget stopped the run.
- `trial-NNNN/`: exact material request, worker response, and simulation log.

Open a saved best fit in Newton's OpenGL viewer to inspect the same selected
experiments, material, fixed scene settings, and simulation duration:

```sh
uv run python scripts/fit_sand.py observe_sand /tmp/sand-fit/best.json
```

To display one experiment, pass its case name:

```sh
uv run python scripts/fit_sand.py observe_sand /tmp/sand-fit/best.json --experiment 1m-5gal-sand-pour
```

Single-experiment views open with a close camera at `(0, -2.32, 1.35)` relative
to the experiment. In the full suite, each Focus button uses the same camera
offset relative to its case.

You can also pass `fit.json` when the stopping tolerance was met. The command
reads its neighboring `inputs.json`; for a fit made with a custom case manifest,
add `--cases PATH/TO/cases.json`. Add `--device cuda:1` to use a different GPU.
The viewer's case controls focus individual experiments.

Capture a case as a 12-second MP4 at 30 FPS. A material record with the same
schema as `best.json` can override the saved fit's material while retaining its
case setup:

```sh
uv run python scripts/render_sand_video.py --best /tmp/sand-fit/best.json --experiment 1m-5gal-sand-pour --output /tmp/sand-pour.mp4
uv run python scripts/render_sand_video.py --best /tmp/sand-fit/best.json --material-config scripts/sand_sysid/configs/sand-literature-and-tuned.json --experiment 1m-5gal-sand-pour --output /tmp/sand-pour-literature.mp4
```

The checked-in `sand-literature-and-tuned.json` records the user-supplied
material values and measured results from six runs. Its material fields are
Young's modulus 19,152,000 Pa, Poisson ratio 0.28, yield pressure 10,000,000 Pa,
damping 0.00376 s, and friction 0.785; other controls retain the previous fit's
fixed values. Its recorded combined error is specific to the saved case settings.

Resume with the same physical settings and seed, optionally extending the total
budget. An interrupted trial is rerun; completed trials are not repeated:

```sh
uv run --with scikit-learn==1.7.2 python scripts/fit_sand.py --output /tmp/sand-fit --resume --max-evaluations 128
```

Start a new search from a valid result in an earlier run. The first candidate
re-simulates its material, then the optimizer explores from that point. Bounds,
cases, scene settings, and frame count must match the earlier run:

```sh
uv run --with scikit-learn==1.7.2 python scripts/fit_sand.py --start-from /tmp/sand-fit/best.json --output /tmp/sand-fit-next --max-evaluations 64 --seed 43
```

Plot candidate error, best-so-far error, and invalid candidates after a run:

```sh
uv run python scripts/plot_sand_history.py /tmp/sand-fit-next
```

The optional optimizer runtime uses scikit-learn 1.7.2, licensed BSD-3-Clause
([license](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING)),
compatible with Newton's Apache-2.0 distribution. It is installed through `uv
--with` and does not add a Newton core dependency. The forest uses the public
[RandomForestRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

Run optimizer and material-assignment regressions:

```sh
uv run --with scikit-learn==1.7.2 python -m unittest scripts.tests.test_sand_forest scripts.tests.test_sand_measurements scripts.tests.test_sand_cases
```

## Funnel simulation

Run the funnel example:

```sh
uv run -m newton.examples mpm_sand_bucket
```

The provisional scene contains 5 US gallons (18.927 L) of bulk sand, a 30 cm
internal bucket diameter, a 20 cm tall funnel, and a 10 cm outlet one meter above
the floor. Sand initially fills the funnel and bucket, at rest; the outlet is
open at time zero. Confirm these assumptions against the experiment before fitting.
This models bulk granular material, not individual physical grains.

### Fixed literature values and continuous fitting

Pass `--fixed-material FILE` with a JSON object such as
`{"young_modulus": 9576000, "poisson_ratio": 0.25}` to remove these values from
optimization. They are recorded in `inputs.json` under `frozen` and assigned to
every candidate. The remaining five search parameters retain their usual bounds.
Pass the same file when resuming; changed fixed values are rejected.

`--continuous` keeps proposing candidates until interrupted, regardless of the
evaluation budget or target error. Checkpoints are saved after each evaluation;
restart with `--resume --continuous` and the original configuration.
