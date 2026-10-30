# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drain five US gallons of dry sand through a funnel into a measured pile.

Height is measured from the funnel outlet to the floor. The outlet is open at
time zero. MPM samples represent bulk sand, not individual physical grains.
All lengths are in meters; the provisional funnel geometry is configurable.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mpm.example_mpm_viscous import Example as ViscousExample
from newton.examples.mpm.sand_material import add_material_arguments, apply_material
from newton.solvers import SolverImplicitMPM


def measure_pile(positions, velocities, radii, *, outlet_height, footprint_fraction=1.0, yaw_degrees=0.0):
    """Measure particle extents along fixed floor axes, including sample radii.

    The optional central quantile footprint excludes scattered samples; 1.0 uses
    every sample. Height is always the maximum surface elevation above z=0.
    Dimensions are provisional until retained_fraction and speed are small.
    All particles have equal mass in this example.
    """
    if not 0 < footprint_fraction <= 1 or not np.isfinite(yaw_degrees):
        raise ValueError("footprint_fraction must be in (0, 1] and yaw_degrees finite")
    if not len(positions) or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Cannot measure empty or non-finite particle states")
    angle = np.radians(yaw_degrees)
    axes = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = positions[:, :2] @ axes
    tail = (1 - footprint_fraction) / 2
    lower = np.quantile(xy - radii[:, None], tail, axis=0)
    upper = np.quantile(xy + radii[:, None], 1 - tail, axis=0)
    return {
        "width_m": float(upper[0] - lower[0]),
        "length_m": float(upper[1] - lower[1]),
        "height_m": float(np.max(positions[:, 2] + radii)),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocities**2, axis=1)))),
        "retained_fraction": float(np.mean(positions[:, 2] >= outlet_height)),
    }


class Example:
    """Simulate a measured sand pour from a funnel onto a rigid floor."""

    def __init__(self, viewer, args):
        """Build the sand particles, funnel, solver, and viewer scene.

        Args:
            viewer: Newton viewer receiving the simulation model.
            args: Parsed pour geometry, material, and timing options.
        """
        for name in (
            "gallons",
            "height",
            "density",
            "voxel_size",
            "fps",
            "substeps",
            "bucket_radius",
            "funnel_height",
            "outlet_diameter",
            "young_modulus",
        ):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if args.outlet_diameter >= 2 * args.bucket_radius:
            raise ValueError("outlet-diameter must be smaller than the bucket diameter")
        if args.outlet_diameter < 4 * args.voxel_size:
            raise ValueError("Resolve the outlet with at least four voxels; reduce --voxel-size")
        for name in ("friction", "dilatancy", "ground_friction", "funnel_friction"):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = args.fps
        self.frame_dt = 1.0 / args.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.volume = args.gallons * 0.003785411784
        self.mass = self.volume * args.density
        self.height = args.height
        self.voxel_size = args.voxel_size
        self.bucket_radius = args.bucket_radius
        aperture = args.outlet_diameter / 2
        funnel_volume = (
            np.pi * args.funnel_height / 3 * (self.bucket_radius**2 + self.bucket_radius * aperture + aperture**2)
        )
        # Solve the fill level, including sand initially inside the funnel.
        if self.volume >= funnel_volume:
            self.fill_depth = args.funnel_height + (self.volume - funnel_volume) / (np.pi * self.bucket_radius**2)
        else:
            lo, hi = 0.0, args.funnel_height
            for _ in range(50):
                mid = (lo + hi) / 2
                radius_mid = aperture + (self.bucket_radius - aperture) * mid / args.funnel_height
                volume_mid = np.pi * mid / 3 * (aperture**2 + aperture * radius_mid + radius_mid**2)
                if volume_mid < self.volume:
                    lo = mid
                else:
                    hi = mid
            self.fill_depth = (lo + hi) / 2

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        spacing = args.voxel_size / 2.0
        xy = np.arange(-self.bucket_radius + spacing / 2, self.bucket_radius, spacing)
        z = np.arange(spacing / 2, self.fill_depth, spacing)
        points = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3)
        local_radius = aperture + (self.bucket_radius - aperture) * np.minimum(points[:, 2] / args.funnel_height, 1.0)
        points = points[np.linalg.norm(points[:, :2], axis=1) < local_radius - spacing]
        if not len(points):
            raise ValueError("voxel-size is too large for the sand volume")
        rng = np.random.default_rng(42)
        points += rng.uniform(-0.2 * spacing, 0.2 * spacing, points.shape)
        points[:, 2] += args.height
        # Assign exact bulk volume regardless of discretization at the cylinder boundary.
        radius = 0.5 * np.cbrt(self.volume / len(points))
        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[self.mass / len(points)] * len(points),
            radius=[radius] * len(points),
        )

        # Reuse the existing thick-walled funnel mesh; the upper section is cylindrical.
        cfg = newton.ModelBuilder.ShapeConfig(mu=args.funnel_friction, density=0.0)
        for bottom_radius, top_radius, depth, bottom_z in (
            (aperture, self.bucket_radius, args.funnel_height, args.height),
            (
                self.bucket_radius,
                self.bucket_radius,
                max(self.fill_depth - args.funnel_height, 0.0) + 0.08,
                args.height + args.funnel_height,
            ),
        ):
            vertices, indices = ViscousExample.create_funnel_mesh(
                bottom_radius, top_radius, depth, bottom_z, thickness=0.012
            )
            builder.add_shape_mesh(
                body=-1,
                mesh=newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False),
                cfg=cfg,
                color=(0.16, 0.38, 0.55),
            )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.ground_friction))
        self.builder = builder
        self.model = builder.finalize()
        self.model.set_gravity((0, 0, -9.81))
        apply_material(self.model, args)
        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size
        config.max_iterations = 100
        config.max_active_cell_count = max(8192, len(points) // 4)
        self.solver = SolverImplicitMPM(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        colors = rng.uniform(0.8, 1.0, (len(points), 1)) * np.array([0.82, 0.64, 0.36])
        self.colors = wp.array(colors, dtype=wp.vec3)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(wp.vec3(2.0, -2.6, 1.65), pitch=-18.0, yaw=128.0)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.metrics = None
        self.use_cuda_graph = (
            self.model.device.is_cuda
            and wp.is_conditional_graph_supported()
            and wp.is_mempool_enabled(self.model.device)
            and not args.no_cuda_graph
        )
        self.graphs = {}
        self.warmed = False

    def step(self):
        """Advance one frame and update pile metrics once per simulated second."""
        for _ in range(self.sim_substeps):
            key = self.state_0.particle_q.ptr
            if self.use_cuda_graph and self.warmed:
                if key not in self.graphs:
                    with wp.ScopedCapture(device=self.model.device) as capture:
                        self._substep()
                    self.graphs[key] = capture.graph
                wp.capture_launch(self.graphs[key])
            else:
                self._substep()
                self.warmed = True
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.check_sparse_grid_rebuild_status()
        self.sim_time += self.frame_dt
        if round(self.sim_time / self.frame_dt) % self.fps == 0:
            self.metrics = self.measure()

    def _substep(self):
        """Advance one MPM substep and project particles outside rigid shapes."""
        self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
        self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)

    def render(self):
        """Send the current particle state to the configured viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "sand", points=self.state_0.particle_q, radii=self.model.particle_radius, colors=self.colors
        )
        self.viewer.end_frame()

    def measure(self, *, footprint_fraction=1.0, yaw_degrees=0.0):
        """Measure the current pile dimensions and motion.

        Args:
            footprint_fraction: Central fraction of samples used for width and length.
            yaw_degrees: Rotation of the measurement axes [degrees].

        Returns:
            Pile dimensions [m], RMS speed [m/s], and retained fraction.
        """
        return measure_pile(
            self.state_0.particle_q.numpy(),
            self.state_0.particle_qd.numpy(),
            self.model.particle_radius.numpy(),
            outlet_height=self.height,
            footprint_fraction=footprint_fraction,
            yaw_degrees=yaw_degrees,
        )

    def render_ui(self, imgui):
        """Show pile measurements and settling status in the viewer panel."""
        imgui.text("Pile dimensions (fixed X/Y axes)")
        if self.metrics is not None:
            for key in ("width_m", "length_m", "height_m", "rms_speed_m_s", "retained_fraction"):
                imgui.text(f"{key}: {self.metrics[key]:.4f}")
            imgui.text("Final dimensions valid after the bucket empties and sand settles.")

    def test_final(self):
        """Verify conservation and finite states without requiring a fixed drain time."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        assert np.isfinite(positions).all() and np.isfinite(velocities).all()
        np.testing.assert_allclose(self.model.particle_mass.numpy().sum(), self.mass, rtol=1e-5)
        represented_volume = np.sum((2 * self.model.particle_radius.numpy()) ** 3)
        np.testing.assert_allclose(represented_volume, self.volume, rtol=1e-5)
        assert positions[:, 2].min() > -self.voxel_size

    @staticmethod
    def create_parser():
        """Create the CLI parser for pour geometry, material, and timing."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--gallons", type=float, default=5.0, help="Bulk sand volume [US gallons]")
        parser.add_argument("--height", type=float, default=1.0, help="Funnel outlet height above floor [m]")
        parser.add_argument("--density", type=float, default=1600.0, help="Bulk sand density [kg/m^3]")
        parser.add_argument("--voxel-size", type=float, default=0.02, help="MPM grid spacing [m]")
        parser.add_argument("--bucket-radius", type=float, default=0.15)
        parser.add_argument("--funnel-height", type=float, default=0.20)
        parser.add_argument("--outlet-diameter", type=float, default=0.10)
        parser.add_argument("--friction", type=float, default=0.65)
        parser.add_argument("--young-modulus", type=float, default=1.0e7)
        parser.add_argument("--dilatancy", type=float, default=0.0)
        parser.add_argument("--ground-friction", type=float, default=0.6)
        parser.add_argument("--funnel-friction", type=float, default=0.25)
        parser.add_argument("--fps", type=int, default=60)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--no-cuda-graph", action="store_true", help="Use eager stepping for comparisons")
        add_material_arguments(parser)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
