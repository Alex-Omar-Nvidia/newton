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

# Quaternions that lay a Z-aligned capsule along the world X and Y axes.
_Q_ALONG_X = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
_Q_ALONG_Y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi)


class Example:
    def __init__(self, viewer, args=None, num_levels: int = 6, solver_type: str = "vbd"):
        self.viewer = viewer
        self.num_levels = num_levels
        self.solver_type = solver_type

        # Simulation cadence.
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        # The deadzone needs to sit between AL's residual convergence error
        # and per-substep gravity displacement. With warm-start, 5 iters has
        # residual ~1e-5 m — same scale as the 1e-5 deadzone — so noise
        # leaks through over time and the stack drifts apart. 25 iters
        # drives residual well below the threshold so the deadzone reliably
        # absorbs it.
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
            # The default gamma=0.999 keeps warm-started lambdas indefinitely
            # so removing a support body leaves the rest floating; gamma=0.5
            # bleeds half each step. Lower gamma also damps the small error
            # AL accumulates each step instead of compounding it — important
            # for long-term rest stability.
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=self.sim_iterations,
                rigid_contact_history=True,
                rigid_avbd_gamma=0.5,
            )
            # The deadzone has to sit between AL's residual noise (~few µm
            # at 25 iterations) and the per-substep displacement on a body
            # whose support has been removed (~1.6e-5 m on the first frame
            # after removal, before warm-started lambdas decay). 1.5e-5 sits
            # in that narrow window, suppressing residual noise while still
            # letting the unbalanced load break the freeze.
            self.solver.rigid_contact_stick_freeze_translation_eps = 1.5e-5
            self.solver.rigid_contact_stick_freeze_angular_eps = 1.5e-5
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        assert np.isfinite(body_q).all(), "Non-finite body transforms"
        assert np.isfinite(body_qd).all(), "Non-finite body velocities"

        # Stack should have settled — both linear and angular velocities small.
        max_speed = float(np.max(np.linalg.norm(body_qd[:, 3:6], axis=1)))
        max_omega = float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1)))
        assert max_speed < 0.1, f"Stack not at rest: max linear speed {max_speed:.4f} m/s"
        assert max_omega < 0.5, f"Stack not at rest: max angular speed {max_omega:.4f} rad/s"

        z_step = 2.0 * CAPSULE_RADIUS
        xy_tol = 0.05  # in-plane drift [m]
        z_tol = 0.5 * CAPSULE_RADIUS

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


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-levels",
        type=int,
        default=6,
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
