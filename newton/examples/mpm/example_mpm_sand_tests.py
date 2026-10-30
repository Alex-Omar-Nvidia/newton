# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run six measured sand experiments together with independent physics and one viewer.

Each case owns its MPM state and contact solver. Display offsets never enter its
physics, so neighboring tests cannot exchange particles, contacts, or grid forces.
Measured data comes from the repository's scripts/sand_sysid/cases.json catalog.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.examples.mpm.example_mpm_bottle_drop import Example as BottleExample
from newton.examples.mpm.example_mpm_sand_bucket import Example as PourExample
from newton.examples.mpm.sand_material import DEFAULTS, FROZEN, add_material_arguments
from newton.solvers import SolverImplicitMPM
from scripts.sand_sysid.cases import DEFAULT_CASES, dimension_error, load_cases


@wp.kernel
def _place_particles(source: wp.array[wp.vec3], destination: wp.array[wp.vec3], start: int, offset: wp.vec3):
    """Copy one case's particles into its translated display tile."""
    i = wp.tid()
    destination[start + i] = source[i] + offset


@wp.kernel
def _place_bodies(source: wp.array[wp.transform], destination: wp.array[wp.transform], start: int, offset: wp.vec3):
    """Copy one case's rigid poses into its translated display tile."""
    i = wp.tid()
    q = source[i]
    destination[start + i] = wp.transform(wp.transform_get_translation(q) + offset, wp.transform_get_rotation(q))


class Example:
    """Display independent sand cases while sharing a simulation clock."""

    def __init__(self, viewer, args):
        """Build selected cases and their combined display model.

        Args:
            viewer: Newton viewer for the combined case layout.
            args: Parsed case selection, scene, material, and timing options.
        """
        if args.gap < 2.0 or not np.isfinite(args.gap):
            raise ValueError("gap must be at least 2 m")
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.frame_dt = 1 / args.fps
        catalog = load_cases(args.cases)
        if args.case:
            missing = set(args.case) - set(catalog)
            if missing:
                raise ValueError(f"Unknown cases: {missing}")
            catalog = {name: value for name, value in catalog.items() if name in args.case}
        self.cases = list(catalog.values())
        self.children = []
        self.offsets = []
        self.particle_starts = []
        self.body_starts = []
        self.resolved = {}
        self.results = {}
        self.overall_error = None
        self._reported = False
        self.tile_size = max(1.2, args.bed_width + 0.05, args.bed_length + 0.05)
        pitch = self.tile_size + args.gap
        columns = min(2, len(self.cases))
        rows = int(np.ceil(len(self.cases) / columns))
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        for index, case in enumerate(self.cases):
            is_bottle = case["test_type"] == "bottle_drop"
            constructor = BottleExample if is_bottle else PourExample
            options = constructor.create_parser().parse_args([])
            for key in ("fps", "substeps", "friction", "young_modulus", "dilatancy", "density", "ground_friction"):
                setattr(options, key, getattr(args, key))
            for key in (*DEFAULTS, *FROZEN):
                setattr(options, key, getattr(args, key))
            config = case["configuration"]
            options.height = config["drop_height_m"]
            options.no_cuda_graph = args.no_cuda_graph
            if is_bottle:
                for key in (
                    "bed_width",
                    "bed_length",
                    "bed_depth",
                    "cap_radius",
                    "cap_height",
                    "bottle_friction",
                    "pre_settle_seconds",
                    "impact_settle_seconds",
                    "lift_speed",
                    "lift_distance",
                    "crater_threshold",
                ):
                    setattr(options, key, getattr(args, key))
                options.water_volume_ml = config["water_volume_ml"]
                options.voxel_size = args.bottle_voxel_size
                options.bottle_radius = (
                    args.bottle_1l_radius if options.water_volume_ml == 1000 else args.bottle_500ml_radius
                )
                options.bottle_height = (
                    args.bottle_1l_height if options.water_volume_ml == 1000 else args.bottle_500ml_height
                )
                for config_key, option_key in (
                    ("sand_mound_width_m", "bed_width"),
                    ("sand_mound_length_m", "bed_length"),
                    ("sand_mound_depth_m", "bed_depth"),
                    ("sand_bulk_density_kg_m3", "density"),
                    ("bottle_height_m", "bottle_height"),
                    ("cap_height_m", "cap_height"),
                ):
                    if config.get(config_key) is not None:
                        setattr(options, option_key, config[config_key])
                for config_key, option_key in (
                    ("bottle_diameter_m", "bottle_radius"),
                    ("cap_diameter_m", "cap_radius"),
                ):
                    if config.get(config_key) is not None:
                        setattr(options, option_key, config[config_key] / 2)
            else:
                options.voxel_size = args.pour_voxel_size
                options.gallons = config["sand_volume_us_gallons"]
                for key in ("outlet_diameter", "funnel_height", "bucket_radius", "funnel_friction"):
                    setattr(options, key, getattr(args, key))
            print(f"Building {case['name']}", flush=True)
            child = constructor(newton.viewer.ViewerNull(), options)
            self.children.append(child)
            self.resolved[case["name"]] = vars(options).copy()
            offset = wp.vec3(
                (index % columns - (columns - 1) / 2) * pitch, (index // columns - (rows - 1) / 2) * pitch, 0
            )
            self.offsets.append(offset)
            self.particle_starts.append(builder.particle_count)
            self.body_starts.append(builder.body_count)
            builder.add_world(
                child.builder, xform=wp.transform(offset, wp.quat_identity()), label_prefix=case["name"] + "/"
            )
        # One displayed floor prevents coincident planes from flickering.
        for i, geo in enumerate(builder.shape_type):
            if geo == newton.GeoType.PLANE:
                builder.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.colors = wp.array(np.concatenate([child.colors.numpy() for child in self.children]), dtype=wp.vec3)
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer.show_particles = False
        if len(self.cases) == 1:
            self._focus_camera(0)
        else:
            self._overview_camera()
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.render_ui, position="side")
        starts, ends = [], []
        for offset in self.offsets:
            x, y, _ = offset
            h = self.tile_size / 2
            corners = [(x - h, y - h, 0.005), (x + h, y - h, 0.005), (x + h, y + h, 0.005), (x - h, y + h, 0.005)]
            for i in range(4):
                starts.append(corners[i])
                ends.append(corners[(i + 1) % 4])
        self.border_starts = wp.array(starts, dtype=wp.vec3)
        self.border_ends = wp.array(ends, dtype=wp.vec3)
        self.border_colors = wp.full(len(starts), wp.vec3(0.3, 0.55, 0.65), dtype=wp.vec3)
        self._sync_display()

    def _focus_camera(self, index):
        """Aim the viewer at one case using its display offset."""
        offset = self.offsets[index]
        self.viewer.set_camera(offset + wp.vec3(0, -2.32, 1.35), pitch=-20, yaw=90)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.look_at(offset + wp.vec3(0, 0, 0.65))

    def _overview_camera(self):
        """Frame every selected experiment in the viewer."""
        self.viewer.set_camera(wp.vec3(8, -11, 10), pitch=-36, yaw=125)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.look_at(wp.vec3(0, 0, 0.4))

    def _sync_display(self):
        """Copy independent case states into the combined display state."""
        for i, child in enumerate(self.children):
            wp.launch(
                _place_particles,
                dim=child.model.particle_count,
                inputs=[child.state_0.particle_q, self.state_0.particle_q, self.particle_starts[i], self.offsets[i]],
            )
            if child.model.body_count:
                wp.launch(
                    _place_bodies,
                    dim=child.model.body_count,
                    inputs=[child.state_0.body_q, self.state_0.body_q, self.body_starts[i], self.offsets[i]],
                )

    def step(self):
        """Advance each case once, then update measurements and final results."""
        for child in self.children:
            child.step()
        self.sim_time += self.frame_dt
        if round(self.sim_time / self.frame_dt) % self.args.fps == 0:
            self.measure()
            if self.args.viewer == "null":
                print(
                    f"t={self.sim_time:.1f}s: "
                    + ", ".join(f"{name}: {result['phase']}" for name, result in self.results.items()),
                    flush=True,
                )
        if not self._reported and self.sim_time >= self.args.num_frames * self.frame_dt - 1e-7:
            self.save_results()
            self._reported = True

    def measure(self):
        """Measure selected cases and combine valid relative RMS errors."""
        errors = []
        for case, child in zip(self.cases, self.children, strict=True):
            measurement = child.measure()
            if case["test_type"] == "sand_pour":
                measurement["valid"] = (
                    self.sim_time >= 8.0 - 1e-8
                    and measurement["retained_fraction"] <= 0.001
                    and measurement["rms_speed_m_s"] < 0.03
                )
                measurement["phase"] = "settled pile" if measurement["valid"] else "pouring / settling"
            record = {
                "measurement": measurement,
                "phase": measurement["phase"],
                "valid": measurement["valid"],
                "target": dict(zip(case["dimensions"], case["target"], strict=True)),
                "error": None,
            }
            if measurement["valid"]:
                residuals, error = dimension_error(measurement, case)
                record.update(residuals=residuals, error=error)
                errors.append(error)
            self.results[case["name"]] = record
        self.overall_error = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) == len(self.cases) else None
        return self.results

    def save_results(self):
        """Write per-case measurements and the combined error as JSON."""
        self.measure()
        result = {
            "time_s": self.sim_time,
            "overall_error": self.overall_error,
            "cases": self.results,
            "resolved_simulation_settings": self.resolved,
            "assumptions": [
                "Unspecified geometry uses configurable provisional defaults.",
                "Bottle is a rigid water-filled mass without slosh or plastic deformation.",
                "Bottle is extracted vertically at constant speed after the impact interval.",
                "Drop heights are cap-to-sand or outlet-to-ground distances.",
                "Crater depth is post-removal rim elevation minus crater floor elevation.",
            ],
        }
        if self.args.results:
            path = Path(self.args.results)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(f"Saved {path}", flush=True)
        print(f"Combined normalized RMS error: {self.overall_error}", flush=True)

    def render(self):
        """Render the combined display state and case boundaries."""
        self._sync_display()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "sand", points=self.state_0.particle_q, radii=self.model.particle_radius, colors=self.colors
        )
        self.viewer.log_lines(
            "case boundaries", starts=self.border_starts, ends=self.border_ends, colors=self.border_colors
        )
        self.viewer.end_frame()

    def render_ui(self, imgui):
        """Show case controls, measured dimensions, and errors in the viewer."""
        imgui.text(f"Time: {self.sim_time:.2f} s | Clear gap: {self.args.gap:g} m")
        imgui.text("Rows: pours, 1 m bottles, 2 m bottles. Left bottle: 1 L; right: 500 mL.")
        imgui.text("Bottle and mound geometry are provisional.")
        for i, case in enumerate(self.cases):
            imgui.separator()
            if imgui.button(f"Focus {case['name']}"):
                self._focus_camera(i)
            result = self.results.get(case["name"])
            if result:
                imgui.text(result["phase"])
                for dimension in case["dimensions"]:
                    value = result["measurement"].get(dimension)
                    if value is not None:
                        imgui.text(f"{dimension}: {value:.3f} / measured {result['target'][dimension]:.3f}")
                if result["error"] is not None:
                    imgui.text(f"Error: {result['error']:.4f}")
        if self.overall_error is not None:
            imgui.text(f"Overall error: {self.overall_error:.4f}")
        if imgui.button("Reset camera" if len(self.cases) == 1 else "Overview"):
            if len(self.cases) == 1:
                self._focus_camera(0)
            else:
                self._overview_camera()

    def test_final(self):
        """Validate independent case states, clock alignment, and ready final metrics."""
        for child in self.children:
            child.test_final()
            np.testing.assert_allclose(child.sim_time, self.sim_time, atol=1e-6)
        assert len({child.state_0.particle_q.ptr for child in self.children}) == len(self.children)
        if self.sim_time >= 8.0 - 1e-8:
            self.measure()
            assert self.overall_error is not None, "Not all cases produced valid final measurements"

    @staticmethod
    def create_parser():
        """Create the CLI parser for case selection and shared scene settings."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
        parser.add_argument("--case", action="append", help="Select cases by name; default is all six")
        parser.add_argument("--results", type=str, default=None, help="Optional JSON results path")
        for name, default in (
            ("gap", 2.0),
            ("bottle-voxel-size", 0.015),
            ("pour-voxel-size", 0.02),
            ("bed-width", 0.45),
            ("bed-length", 0.45),
            ("bed-depth", 0.30),
            ("bottle-1l-radius", 0.04),
            ("bottle-1l-height", 0.25),
            ("bottle-500ml-radius", 0.033),
            ("bottle-500ml-height", 0.20),
            ("cap-radius", 0.015),
            ("cap-height", 0.015),
            ("bottle-friction", 0.3),
            ("pre-settle-seconds", 0.5),
            ("impact-settle-seconds", 3.0),
            ("lift-speed", 0.25),
            ("lift-distance", 0.6),
            ("crater-threshold", 0.003),
            ("friction", 0.65),
            ("young-modulus", 1.0e7),
            ("dilatancy", 0.0),
            ("density", 1600.0),
            ("ground-friction", 0.6),
            ("funnel-friction", 0.25),
            ("outlet-diameter", 0.10),
            ("funnel-height", 0.20),
            ("bucket-radius", 0.15),
        ):
            parser.add_argument(f"--{name}", type=float, default=default)
        parser.add_argument("--fps", type=int, default=60)
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--no-cuda-graph", action="store_true", help="Use eager stepping for comparisons")
        add_material_arguments(parser)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
