# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Capsule Stack
#
# Lincoln-log style dense stack of capsules. Each level holds a pair of
# parallel capsules; consecutive levels alternate between the X and Y
# axes so each pair rests across the pair below it.
#
# The default VBD/AVBD solver should drive the stack to a stable rest
# with minimal drift. Pass ``--solver xpbd`` to verify the impulse-based
# Gauss-Seidel solver converges to the same rest pose on the same scene.
#
# Command: python -m newton.examples capsule_stack
# With XPBD: python -m newton.examples capsule_stack --solver xpbd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

CAPSULE_RADIUS = 0.1
CAPSULE_HALF_HEIGHT = 0.5  # cylindrical half-length (excluding hemispherical caps)
PAIR_HALF_SEPARATION = 0.45  # half the in-plane distance between the two capsules of a level
SETTLE_TEST_WINDOW = 30
SETTLE_MAX_SPEED = 0.2
SETTLE_MAX_OMEGA = 0.5
SETTLE_MAX_MEAN_SPEED = 0.15
SETTLE_MAX_MEAN_OMEGA = 0.25
SETTLE_MAX_WINDOW_MOTION = 0.03
SUPPORT_REMOVAL_TEST_FRAMES = 24
SUPPORT_REMOVAL_MIN_DROP = 0.015

# Quaternions that lay a Z-aligned capsule along the world X and Y axes.
_Q_ALONG_X = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
_Q_ALONG_Y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi)


class Example:
    def __init__(self, viewer, args=None, num_levels: int = 10, solver_type: str = "vbd"):
        self.viewer = viewer
        self.num_levels = num_levels
        self.solver_type = solver_type

        # Simulation cadence.
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 120
        self.sim_iterations = 25
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()

        # VBD prefers stiff, lightly damped contact for rigids; XPBD's
        # impulse-based Gauss-Seidel relaxation handles either regime.
        builder.default_shape_cfg.ke = 1.0e6
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 0.6

        builder.add_ground_plane()

        # The level-0 capsules rest on the ground with their centers at z = r.
        # Each subsequent level sits on the level below across two contact
        # points, lifting the next center by 2 * r.
        z_step = 2.0 * CAPSULE_RADIUS
        z_base = CAPSULE_RADIUS

        self.capsule_bodies: list[int] = []
        self.initial_positions: list[wp.vec3] = []
        for level in range(self.num_levels):
            along_x = (level % 2) == 0
            q = _Q_ALONG_X if along_x else _Q_ALONG_Y
            z = z_base + level * z_step

            for sign in (-1.0, +1.0):
                if along_x:
                    pos = wp.vec3(0.0, sign * PAIR_HALF_SEPARATION, z)
                else:
                    pos = wp.vec3(sign * PAIR_HALF_SEPARATION, 0.0, z)

                body = builder.add_body(
                    xform=wp.transform(pos, q),
                    label=f"capsule_l{level}_{'+' if sign > 0 else '-'}",
                )
                builder.add_shape_capsule(
                    body,
                    radius=CAPSULE_RADIUS,
                    half_height=CAPSULE_HALF_HEIGHT,
                )
                self.capsule_bodies.append(body)
                self.initial_positions.append(pos)

        if self.solver_type == "vbd":
            builder.color()

        self.model = builder.finalize()

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=self.sim_iterations,
                rigid_contact_history=False,
            )
            stick_freeze_eps = 0.0
            self.solver.rigid_contact_stick_freeze_translation_eps = stick_freeze_eps
            self.solver.rigid_contact_stick_freeze_angular_eps = stick_freeze_eps
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=self.sim_iterations,
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type!r}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._body_masses = self.model.body_mass.numpy().copy()
        self._bottom_layer_bodies = self.capsule_bodies[:2]
        self._top_layer_bodies = self.capsule_bodies[-2:]
        self._previous_bottom_momentum: np.ndarray | None = None
        self._previous_top_momentum: np.ndarray | None = None
        self._latest_bottom_force = np.zeros(3, dtype=np.float32)
        self._latest_top_force = np.zeros(3, dtype=np.float32)
        self._test_body_q_window: list[np.ndarray] = []
        self._test_max_speed_window: list[float] = []
        self._test_max_omega_window: list[float] = []

        pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.5, -2.5, 1.2),
            pitch=-15.0,
            yaw=135.0,
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self._log_layer_forces()

    def _layer_momentum(self, body_qd: np.ndarray, bodies: list[int]) -> np.ndarray:
        masses = self._body_masses[bodies, None]
        velocities = body_qd[bodies, 3:6]
        return np.sum(masses * velocities, axis=0)

    def _log_layer_forces(self):
        body_qd = self.state_0.body_qd.numpy()
        bottom_momentum = self._layer_momentum(body_qd, self._bottom_layer_bodies)
        top_momentum = self._layer_momentum(body_qd, self._top_layer_bodies)

        if self._previous_bottom_momentum is None or self._previous_top_momentum is None:
            bottom_force = np.zeros(3, dtype=np.float32)
            top_force = np.zeros(3, dtype=np.float32)
        else:
            bottom_force = (bottom_momentum - self._previous_bottom_momentum) / self.frame_dt
            top_force = (top_momentum - self._previous_top_momentum) / self.frame_dt

        self._previous_bottom_momentum = bottom_momentum
        self._previous_top_momentum = top_momentum
        self._latest_bottom_force = np.asarray(bottom_force, dtype=np.float32)
        self._latest_top_force = np.asarray(top_force, dtype=np.float32)

        self.viewer.log_scalar("Bottom layer |net force| [N]", np.linalg.norm(self._latest_bottom_force), smoothing=2)
        self.viewer.log_scalar("Bottom layer net Fz [N]", self._latest_bottom_force[2], smoothing=2)
        self.viewer.log_scalar("Top layer |net force| [N]", np.linalg.norm(self._latest_top_force), smoothing=2)
        self.viewer.log_scalar("Top layer net Fz [N]", self._latest_top_force[2], smoothing=2)

    def gui(self, ui):
        ui.text(f"Levels: {self.num_levels}")
        ui.text(f"Bottom |F_net|: {np.linalg.norm(self._latest_bottom_force):.3f} N")
        ui.text(f"Bottom Fz: {self._latest_bottom_force[2]:.3f} N")
        ui.text(f"Top |F_net|: {np.linalg.norm(self._latest_top_force):.3f} N")
        ui.text(f"Top Fz: {self._latest_top_force[2]:.3f} N")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        assert np.isfinite(body_q).all(), "Non-finite body transforms"
        assert np.isfinite(body_qd).all(), "Non-finite body velocities"

        max_speed = float(np.max(np.linalg.norm(body_qd[:, 3:6], axis=1)))
        max_omega = float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1)))
        self._test_body_q_window.append(body_q.copy())
        self._test_max_speed_window.append(max_speed)
        self._test_max_omega_window.append(max_omega)

        if len(self._test_body_q_window) > SETTLE_TEST_WINDOW:
            self._test_body_q_window.pop(0)
            self._test_max_speed_window.pop(0)
            self._test_max_omega_window.pop(0)

    def test_final(self):
        if not self._test_body_q_window:
            self.test_post_step()

        body_q = self._test_body_q_window[-1]
        body_qd = self.state_0.body_qd.numpy()

        assert np.isfinite(body_q).all(), "Non-finite body transforms"
        assert np.isfinite(body_qd).all(), "Non-finite body velocities"

        max_speed = float(np.max(np.linalg.norm(body_qd[:, 3:6], axis=1)))
        max_omega = float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1)))
        assert max_speed < SETTLE_MAX_SPEED, f"Stack not at rest: max linear speed {max_speed:.4f} m/s"
        assert max_omega < SETTLE_MAX_OMEGA, f"Stack not at rest: max angular speed {max_omega:.4f} rad/s"

        mean_window_speed = float(np.mean(self._test_max_speed_window))
        mean_window_omega = float(np.mean(self._test_max_omega_window))
        assert mean_window_speed < SETTLE_MAX_MEAN_SPEED, (
            f"Stack not settled over final {len(self._test_max_speed_window)} frames: "
            f"mean linear speed {mean_window_speed:.4f} m/s"
        )
        assert mean_window_omega < SETTLE_MAX_MEAN_OMEGA, (
            f"Stack not settled over final {len(self._test_max_omega_window)} frames: "
            f"mean angular speed {mean_window_omega:.4f} rad/s"
        )

        window_positions = np.stack([q[:, :3] for q in self._test_body_q_window])
        max_window_motion = float(np.max(np.linalg.norm(window_positions - window_positions[-1], axis=2)))
        assert max_window_motion < SETTLE_MAX_WINDOW_MOTION, (
            f"Stack drifting over final {len(self._test_body_q_window)} frames: "
            f"max body displacement {max_window_motion:.4f} m"
        )

        z_step = 2.0 * CAPSULE_RADIUS
        xy_tol = 0.03  # in-plane drift [m]
        z_tol = CAPSULE_RADIUS

        for body, init_pos in zip(self.capsule_bodies, self.initial_positions, strict=True):
            pos = body_q[body, :3]
            xy_drift = float(np.linalg.norm(pos[:2] - np.array([init_pos[0], init_pos[1]])))
            assert xy_drift < xy_tol, f"Capsule body {body} drifted {xy_drift:.4f} m in XY (max {xy_tol:.4f} m)"
            assert abs(pos[2] - init_pos[2]) < z_tol, (
                f"Capsule body {body} z={pos[2]:.4f} m, expected near {init_pos[2]:.4f} m"
            )

        # Sanity: each level's average z should match its expected resting height.
        for level in range(self.num_levels):
            expected_z = CAPSULE_RADIUS + level * z_step
            level_z = body_q[2 * level : 2 * level + 2, 2].mean()
            assert abs(level_z - expected_z) < z_tol, (
                f"Level {level} mean z={level_z:.4f} m, expected {expected_z:.4f} m"
            )

        self._test_support_removal_falls(body_qd)

    def _test_support_removal_falls(self, settled_body_qd):
        if self.num_levels < 2:
            return

        body_q = self._test_body_q_window[-1].copy()
        body_qd = settled_body_qd.copy()
        falling_bodies = self.capsule_bodies[2:]
        initial_falling_z = float(np.mean(body_q[falling_bodies, 2]))

        for offset, body in enumerate(self.capsule_bodies[:2]):
            body_q[body, 0] += 100.0 + offset
            body_q[body, 1] += 100.0
            body_q[body, 2] = CAPSULE_RADIUS
            body_qd[body, :] = 0.0

        self.state_0.body_q.assign(body_q)
        self.state_0.body_qd.assign(body_qd)
        self.state_1.body_q.assign(body_q)
        self.state_1.body_qd.assign(body_qd)
        if hasattr(self.solver, "body_q_prev"):
            self.solver.body_q_prev.assign(body_q)
        self.graph = None

        min_falling_z = initial_falling_z
        for _ in range(SUPPORT_REMOVAL_TEST_FRAMES):
            self.step()
            removal_body_q = self.state_0.body_q.numpy()
            min_falling_z = min(min_falling_z, float(np.mean(removal_body_q[falling_bodies, 2])))

        drop = initial_falling_z - min_falling_z
        assert drop > SUPPORT_REMOVAL_MIN_DROP, (
            f"Stack did not fall after bottom support removal: mean z dropped {drop:.4f} m "
            f"over {SUPPORT_REMOVAL_TEST_FRAMES} frames"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-levels",
        type=int,
        default=10,
        help="Number of capsule pair levels to stack (alternating X/Y).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="vbd",
        choices=["vbd", "xpbd"],
        help="Solver to use: vbd (default, AVBD for rigids) or xpbd (impulse-based Gauss-Seidel).",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args, num_levels=args.num_levels, solver_type=args.solver)
    newton.examples.run(example, args)
