# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a VBD tire into a 1.0 x 1.0 x 1.0 m MPM gravel pit.

Run with ``python -m newton.examples vbd_mpm_gravel_pit``. The experimental
proxy coupling models gravel as a frictional continuum, not individual stones.
The wheel is held for three simulated seconds. Live plots report gravel
kinetic energy and wheel penetration relative to the bed at release. Material
sliders apply immediately; Restart repeats the experiment with those settings.
Two walls are hidden for a cutaway view but still participate in collisions.
The upright tire is a 30 cm diameter, 10 cm thick solid cylinder.
A red band follows its mass-weighted center height, including when buried.
MPM samples represent continuum volumes rather than individual gravel stones.
The 5 mm samples are drawn at the same width with partial opacity so the buried
tire remains visible. Their centers are spaced at half the MPM grid width.
The loose granular preset permits bed compaction and continued sinking;
this is a visual experiment, not calibrated gravel.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.example_vbd_mpm_coupled_solver import Example as CoupledExample
from newton.examples.multiphysics.example_vbd_mpm_coupled_solver import _gather_particles


@wp.kernel
def _hold_wheel(
    initial: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    positions[i] = initial[i]
    velocities[i] = wp.vec3(0.0)


@wp.kernel
def _point_transforms(points: wp.array[wp.vec3], transforms: wp.array[wp.transform]):
    i = wp.tid()
    transforms[i] = wp.transform(points[i], wp.quat_identity())


class Example(CoupledExample):
    PIT_LENGTH = 1.0
    PIT_WIDTH = 1.0
    PIT_DEPTH = 1.0
    TIRE_DIAMETER = 0.3
    TIRE_WIDTH = 0.1
    SAMPLE_WIDTH = 0.005
    GRAVEL_OPACITY = 0.5

    def __init__(self, viewer, args):
        if not 0.01 <= args.voxel_size <= 0.04:
            raise ValueError("voxel-size must be between 0.01 and 0.04 m")
        if args.settle_seconds < 0.0:
            raise ValueError("settle-seconds must be nonnegative")
        if args.simulation_fps <= 0 or args.simulation_substeps <= 0:
            raise ValueError("simulation-fps and simulation-substeps must be positive")
        if args.mpm_friction < 0.0 or args.mpm_young_modulus <= 0.0 or args.mpm_yield_pressure <= 0.0:
            raise ValueError("MPM friction must be nonnegative; stiffness and yield pressure must be positive")
        self.args = args
        self.mpm_max_active_cell_count = 1 << 18
        self.release_time = args.settle_seconds
        self.released = False
        self.surface_at_release = self.PIT_DEPTH
        self.restart_requested = False
        self.plot_layout_frames = 0
        self.material = getattr(
            self,
            "material",
            {
                "friction": args.mpm_friction,
                "young_modulus": args.mpm_young_modulus,
                "damping": 0.05,
                "yield_pressure": args.mpm_yield_pressure,
                "dilatancy": 0.0,
            },
        )
        super().__init__(viewer, args)
        self.fps = args.simulation_fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.simulation_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.initial_q = wp.clone(self.model.particle_q)
        self.sample_width = self.SAMPLE_WIDTH
        self.sample_spacing = args.voxel_size / 2.0
        self.mpm_render_radii.fill_(self.sample_width / 2.0)
        count = len(self.mpm_particles)
        self.gravel_transforms = wp.empty(count, dtype=wp.transform, device=self.model.device)
        self.gravel_scales = wp.full(count, wp.vec3(self.sample_width / 2.0), dtype=wp.vec3, device=self.model.device)
        self.gravel_opacities = wp.full(count, self.GRAVEL_OPACITY, dtype=float, device=self.model.device)
        # An octahedron keeps the transparent instancing cost modest for the bed.
        self.gravel_mesh_points = wp.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.gravel_mesh_indices = wp.array(
            [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5],
            dtype=int,
            device=self.model.device,
        )
        self.masses = self.model.particle_mass.numpy()
        self.apply_material()
        self.metrics = {}
        self.sample_metrics()
        self.band_local = np.array(
            [
                (x, y, z)
                for z in (-0.003, 0.003)
                for x, y in (
                    (-self.PIT_LENGTH / 2.0 - 0.021, -self.PIT_WIDTH / 2.0 - 0.021),
                    (self.PIT_LENGTH / 2.0 + 0.021, -self.PIT_WIDTH / 2.0 - 0.021),
                    (self.PIT_LENGTH / 2.0 + 0.021, self.PIT_WIDTH / 2.0 + 0.021),
                    (-self.PIT_LENGTH / 2.0 - 0.021, self.PIT_WIDTH / 2.0 + 0.021),
                )
            ],
            dtype=np.float32,
        )
        band_indices = []
        for i in range(4):
            j = (i + 1) % 4
            band_indices.extend((i, j, j + 4, i, j + 4, i + 4))
        self.band_points = wp.empty(8, dtype=wp.vec3, device=self.model.device)
        self.band_indices = wp.array(band_indices, dtype=int, device=self.model.device)
        self.viewer.set_camera(wp.vec3(2.4, -2.6, 1.8), pitch=-21.0, yaw=135.0)
        if not getattr(self, "ui_registered", False) and hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.render_ui, position="side")
            self.ui_registered = True

    def capture(self):
        # The hold/release transition and live material changes use ordinary launches.
        self.graph = None

    def _emit_softbody(self, builder):
        # Map the XZ square to a disk and retain Y as the horizontal axle.
        # Compute rest poses only after this mapping to avoid prestressing the wheel.
        grid = newton.ModelBuilder()
        grid.add_soft_grid(
            pos=wp.vec3(-1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=8,
            dim_y=3,
            dim_z=8,
            cell_x=0.25,
            cell_y=2.0 / 3.0,
            cell_z=0.25,
            density=1000.0,
            k_mu=2.0e4,
            k_lambda=2.0e4,
            k_damp=0.1,
        )
        vertices = np.asarray(grid.particle_q, dtype=np.float64)
        x, y, z = vertices.T.copy()
        vertices[:, 0] = self.TIRE_DIAMETER / 2.0 * x * np.sqrt(1.0 - z * z / 2.0)
        vertices[:, 1] = self.TIRE_WIDTH / 2.0 * y
        vertices[:, 2] = self.TIRE_DIAMETER / 2.0 * z * np.sqrt(1.0 - x * x / 2.0)
        indices = np.asarray(grid.tet_indices)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, self.PIT_DEPTH + self.TIRE_DIAMETER / 2.0 + 0.1),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=vertices.tolist(),
            indices=indices.flatten().tolist(),
            density=1100.0,
            k_mu=2.0e5,
            k_lambda=4.0e5,
            k_damp=100.0,
            particle_radius=0.0025,
            color=wp.vec3(0.9, 0.25, 0.12),
        )
        return list(range(builder.particle_count))

    def _emit_mpm_heap(self, builder, voxel_size):
        # All five box shapes are fixed rigid colliders with interior floor z=0.
        hx = self.PIT_LENGTH / 2.0
        hy = self.PIT_WIDTH / 2.0
        wall_height = self.PIT_DEPTH + 0.1
        walls = [
            ((0, 0, -0.01), (hx + 0.02, hy + 0.02, 0.01), True),
            ((-hx - 0.01, 0, wall_height / 2.0), (0.01, hy + 0.02, wall_height / 2.0), True),
            ((hx + 0.01, 0, wall_height / 2.0), (0.01, hy + 0.02, wall_height / 2.0), False),
            ((0, -hy - 0.01, wall_height / 2.0), (hx, 0.01, wall_height / 2.0), False),
            ((0, hy + 0.01, wall_height / 2.0), (hx, 0.01, wall_height / 2.0), True),
        ]
        for center, half_size, visible in walls:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=half_size[0],
                hy=half_size[1],
                hz=half_size[2],
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6, is_visible=visible),
                color=wp.vec3(0.28, 0.35, 0.40),
            )
        spacing = voxel_size / 2.0
        axes = [
            np.arange(-hx + spacing, hx - spacing / 2.0, spacing),
            np.arange(-hy + spacing, hy - spacing / 2.0, spacing),
            np.arange(spacing, self.PIT_DEPTH, spacing),
        ]
        positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        positions += np.random.default_rng(42).uniform(-0.1, 0.1, positions.shape) * spacing
        start = builder.particle_count
        count = len(positions)
        builder.add_particles(
            pos=positions.tolist(),
            vel=np.zeros_like(positions).tolist(),
            mass=[1800.0 * spacing**3] * count,
            radius=[self.SAMPLE_WIDTH / 2.0] * count,
            custom_attributes={f"mpm:{key}": value for key, value in self.material.items()},
        )
        return list(range(start, builder.particle_count))

    def apply_material(self):
        for key, value in self.material.items():
            getattr(self.model.mpm, key)[self.mpm_particle_ids].fill_(value)
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if not self.released:
                wp.launch(
                    _hold_wheel,
                    dim=len(self.soft_particles),
                    inputs=[self.initial_q, self.state_1.particle_q, self.state_1.particle_qd],
                    device=self.model.device,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.restart_requested:
            self.__init__(self.viewer, self.args)
        if not self.released and self.sim_time + 1.0e-8 >= self.release_time:
            gravel_z = self.state_0.particle_q.numpy()[self.mpm_particles, 2]
            self.surface_at_release = float(np.quantile(gravel_z, 0.999))
            self.released = True
        super().step()
        self.sample_metrics()
        for name, value in self.metrics.items():
            self.viewer.log_scalar(name, value, clear=self.sim_time <= self.frame_dt)

    def sample_metrics(self):
        q = self.state_0.particle_q.numpy()
        v = self.state_0.particle_qd.numpy()[self.mpm_particles]
        mass = self.masses[self.mpm_particles]
        energy = float(0.5 * np.sum(mass * np.sum(v * v, axis=1)))
        bottom = float(q[self.soft_particles, 2].min())
        self.object_height = float(np.average(q[self.soft_particles, 2], weights=self.masses[self.soft_particles]))
        self.metrics = {
            "Gravel kinetic energy [J]": energy,
            "Gravel RMS speed [m/s]": float(np.sqrt(2.0 * energy / mass.sum())),
            "Wheel penetration below release surface [m]": max(0.0, self.surface_at_release - bottom),
            "Wheel bottom clearance [m]": bottom - 0.0025,
        }

    def render(self):
        render_state = newton.examples.get_coupled_view_state(self)
        wp.launch(
            _gather_particles,
            dim=len(self.mpm_particles),
            inputs=[self.mpm_particle_ids, render_state.particle_q, self.mpm_render_points],
            device=self.model.device,
        )
        wp.launch(
            _point_transforms,
            dim=len(self.mpm_particles),
            inputs=[self.mpm_render_points, self.gravel_transforms],
            device=self.model.device,
        )
        band_points = self.band_local.copy()
        band_points[:, 2] += self.object_height
        self.band_points.assign(band_points)
        self.viewer.begin_frame(self.sim_time)
        show_particles = getattr(self.viewer, "show_particles", True)
        self.viewer.show_particles = False
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.show_particles = show_particles
        self.viewer.log_mesh(
            "/gravel_sample_mesh",
            self.gravel_mesh_points,
            self.gravel_mesh_indices,
            hidden=True,
        )
        self.viewer.log_instances(
            "/gravel_samples",
            "/gravel_sample_mesh",
            self.gravel_transforms,
            self.gravel_scales,
            self.mpm_render_colors,
            None,
            hidden=not show_particles,
            opacities=self.gravel_opacities,
        )
        # This is only a visual marker: it has no mass or collision geometry.
        self.viewer.log_mesh(
            "/wheel_height_band",
            self.band_points,
            self.band_indices,
            color=(1.0, 0.02, 0.02),
            backface_culling=False,
            dynamic=True,
        )
        self.viewer.end_frame()

    def render_ui(self, imgui):
        # Bring the live plots on screen even if a previous viewer saved their
        # position on a larger display. The window exists after the first frame.
        if self.plot_layout_frames < 2 and hasattr(imgui, "get_io"):
            size = imgui.get_io().display_size
            imgui.set_window_pos("Plots", imgui.ImVec2(max(0.0, size.x - 410.0), 300.0))
            imgui.set_window_size("Plots", imgui.ImVec2(400.0, min(620.0, size.y - 310.0)))
            self.plot_layout_frames += 1
        imgui.text(f"Time: {self.sim_time:.2f} s | " + ("Wheel released" if self.released else "Settling gravel"))
        imgui.text("Pit: 1.0 x 1.0 m; depth: 1.0 m")
        imgui.text("Tire: diameter 30 cm, width 10 cm")
        imgui.text(f"Mass: {self.masses[self.soft_particles].sum():.3f} kg")
        imgui.text(f"Red band: center at {self.object_height:.3f} m")
        imgui.text(f"MPM sample/display width: {self.sample_width * 1000.0:.2f} mm")
        imgui.text(f"Sample center spacing: {self.sample_spacing * 1000.0:.1f} mm")
        imgui.text(f"MPM grid cell: {self.args.voxel_size * 1000.0:.2f} mm")
        imgui.text("Cutaway: two walls hidden.")
        imgui.text("All wall collisions stay active.")
        imgui.text("Depth reference: bed at release")
        imgui.text("(99.9th-percentile height).")
        imgui.text(f"Floor clearance: {self.metrics['Wheel bottom clearance [m]']:.3f} m")
        sliders = [
            ("friction", "MPM internal friction", 0.0, 1.2, 1.0),
            ("young_modulus", "Stiffness [MPa]", 0.1, 2000.0, 1.0e6),
            ("damping", "Damping [s]", 0.0, 1.0, 1.0),
            ("yield_pressure", "Yield pressure [MPa]", 0.01, 20.0, 1.0e6),
            ("dilatancy", "Dilatancy", 0.0, 1.0, 1.0),
        ]
        changed_any = False
        for key, label, lower, upper, scale in sliders:
            imgui.text(label)
            changed, value = imgui.slider_float(f"##{key}", self.material[key] / scale, lower, upper)
            if changed:
                self.material[key] = value * scale
                changed_any = True
        if changed_any:
            self.apply_material()
        imgui.text("Sliders apply live. Restart to")
        imgui.text("compare from the initial state.")
        if imgui.button("Restart with current material"):
            self.restart_requested = True

    def test_final(self):
        """Verify finite motion, containment, and wheel clearance."""
        q = self.state_0.particle_q.numpy()
        assert np.isfinite(q).all()
        assert np.isfinite(self.state_0.particle_qd.numpy()).all()
        gravel = q[self.mpm_particles]
        tolerance = self.args.voxel_size
        assert gravel[:, 2].min() > -tolerance, "Gravel escaped through the floor"
        assert np.abs(gravel[:, 0]).max() < self.PIT_LENGTH / 2.0 + tolerance, "Gravel escaped through an end wall"
        assert np.abs(gravel[:, 1]).max() < self.PIT_WIDTH / 2.0 + tolerance, "Gravel escaped through a side wall"
        assert self.metrics["Wheel bottom clearance [m]"] > 0.0, "Wheel reached the floor"
        if not self.released:
            np.testing.assert_allclose(q[self.soft_particles], self.initial_q.numpy()[self.soft_particles])

    @staticmethod
    def create_parser():
        parser = CoupledExample.create_parser()
        parser.set_defaults(voxel_size=0.03, collider_thickness=0.0025, mpm_iterations=60, num_frames=1200)
        parser.add_argument(
            "--mpm-friction", type=float, default=0.0625, help="Uniform gravel internal friction coefficient"
        )
        parser.add_argument(
            "--mpm-young-modulus", type=float, default=1.0e9, help="Uniform gravel Young's modulus in Pa"
        )
        parser.add_argument(
            "--mpm-yield-pressure", type=float, default=1.0e7, help="Uniform gravel yield pressure in Pa"
        )
        parser.add_argument("--simulation-fps", type=float, default=60.0, help="Simulation frames per second")
        parser.add_argument("--simulation-substeps", type=int, default=4, help="Solver substeps per simulation frame")
        parser.add_argument(
            "--settle-seconds",
            type=float,
            default=3.0,
            help="Simulated seconds before releasing the 30 cm diameter, 10 cm wide VBD tire",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
