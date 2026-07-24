# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Craig-Bampton ROM
#
# Loads the reduced matrices and surface-recovery data exported by
# example_brick_CB.m, builds a ModalBasis with ModalGeneratorCraigBampton,
# and moves the left interface vertically through a 0.5 m stroke.
# At each end it rolls the interface about X through +15, -15, and back to zero,
# then dwells so the exported damping can settle both interface and
# fixed-interface vibration.
#
# Command:
#   uv run --extra examples python \
#       newton/examples/fea/craig_bampton_ROM_elastic_links/example_craig_bampton_rom.py
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import find_free_joint_q_start, quat_rotate, set_camera_from_bounds
from newton.examples.basic._reduced_elastic_contact import (
    apply_kinematic_targets,
    finite_difference_target_velocities,
)

EXAMPLE_DIR = Path(__file__).resolve().parent


def _load_csv(name: str, *, dtype: type = float) -> np.ndarray:
    """Load a numeric CSV matrix exported by the MATLAB example."""
    return np.loadtxt(EXAMPLE_DIR / name, delimiter=",", dtype=dtype, ndmin=2)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # The lowest exported mode is about 742 Hz. Advance physical time in slow
        # motion so its damped oscillation remains visible at a 60 Hz render rate.
        self.time_scale = 0.05
        self.sim_substeps = 1
        self.sim_dt = self.time_scale * self.frame_dt
        self.sim_time = 0.0
        self.render_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args

        self.bottom_z = 0.25
        self.stroke = 0.5
        self.top_z = self.bottom_z + self.stroke
        self.rotation_amplitude = np.deg2rad(15.0)
        self.rotation_leg_duration = 0.5
        self.settle_duration = 1.0
        self.endpoint_duration = 3.0 * self.rotation_leg_duration + self.settle_duration
        self.travel_duration = 0.75
        self.motion_period = 2.0 * (self.endpoint_duration + self.travel_duration)
        self.default_mass_scale = 3.0
        self.mass_scale = self.default_mass_scale
        self.stiffness_scale = 1.0
        self.damping_scale = 1.0

        points = _load_csv("cb_rom_points.csv").astype(np.float32)
        faces = _load_csv("cb_rom_faces.csv", dtype=int).astype(np.int32) - 1
        interfaces = _load_csv("cb_rom_interfaces.csv")
        mass = _load_csv("cb_rom_mass.csv")
        stiffness = _load_csv("cb_rom_stiffness.csv")
        damping = _load_csv("cb_rom_damping.csv")
        recovery = _load_csv("cb_rom_recovery.csv")

        self.generator = newton.ModalGeneratorCraigBampton(
            interface_positions=interfaces,
            mass_matrix=mass,
            stiffness_matrix=stiffness,
            damping_matrix=damping,
            sample_points=points,
            recovery_matrix=recovery,
            interface_names=("left", "right"),
            label="matlab_brick_craig_bampton",
        )
        self.basis = self.generator.build()
        self.surface_sample_count = points.shape[0]
        internal_start = self.generator.interface_dof_count
        self._internal_recovery = recovery[:, internal_start:]

        initial_mode_q = np.zeros(self.basis.mode_count, dtype=np.float32)

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        inertia = wp.mat33(*self.generator.inertia.astype(np.float32).ravel())
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        self.base = builder.add_body(
            xform=wp.transform(wp.vec3(-0.5, 0.0, self.bottom_z), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            is_kinematic=True,
            label="moving_left_interface",
        )
        builder.add_shape_box(
            self.base,
            xform=wp.transform(wp.vec3(-0.035, 0.0, 0.0), wp.quat_identity()),
            hx=0.035,
            hy=0.13,
            hz=0.13,
            cfg=shape_cfg,
            label="left_interface_fixture",
        )

        self.body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.bottom_z), wp.quat_identity()),
            com=wp.vec3(*self.generator.com.astype(np.float32)),
            inertia=inertia,
            mass=self.generator.mass,
            mode_q=initial_mode_q,
            modal_basis=self.basis,
            lock_inertia=True,
            label="craig_bampton_brick",
        )

        builder.add_shape_mesh(
            self.body,
            mesh=newton.Mesh(points, faces.ravel(), compute_inertia=False),
            cfg=shape_cfg,
            label="craig_bampton_brick_surface",
        )
        builder.add_joint_fixed(
            parent=self.base,
            child=self.body,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            label="left_interface_clamp",
        )
        builder.color()

        self.model = builder.finalize()
        # Render the recovered elastic surface without the generic diagnostic
        # centerline, whose primitive-sized placement is not useful for this mesh.
        self.model.elastic_render_point_total_count = 0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        elastic_joint = int(self.model.elastic_joint.numpy()[0])
        q_start = int(self.model.joint_q_start.numpy()[elastic_joint])
        qd_start = int(self.model.joint_qd_start.numpy()[elastic_joint])
        self.modal_q_slice = slice(q_start + 7, q_start + 7 + self.basis.mode_count)
        base_q_start, base_qd_start = find_free_joint_q_start(self.model, self.base)
        self._body_q_starts = {self.base: base_q_start}
        self._body_qd_starts = {self.base: base_qd_start}
        self._elastic_frame_q_starts = {self.body: q_start}
        self._elastic_frame_qd_starts = {self.body: qd_start}

        self._base_mode_mass = self.model.elastic_mode_mass.numpy().copy()
        self._base_mode_stiffness = self.model.elastic_mode_stiffness.numpy().copy()
        self._base_mode_damping = self.model.elastic_mode_damping.numpy().copy()
        self._base_coupling_linear = self.model.elastic_mode_coupling_linear.numpy().copy()
        self._base_coupling_angular = self.model.elastic_mode_coupling_angular.numpy().copy()
        self._base_coupling_centrifugal = self.model.elastic_mode_coupling_centrifugal.numpy().copy()
        self._base_coupling_coriolis = self.model.elastic_mode_coupling_coriolis.numpy().copy()
        self._base_body_mass = self.model.body_mass.numpy().copy()
        self._base_body_inv_mass = self.model.body_inv_mass.numpy().copy()
        self._base_body_inertia = self.model.body_inertia.numpy().copy()
        self._base_body_inv_inertia = self.model.body_inv_inertia.numpy().copy()

        self.max_mode_norm = 0.0
        self.final_mode_norm = 0.0
        self.max_vertex_displacement = 0.0
        self.final_vertex_displacement = 0.0
        self.max_internal_coordinate_norm = 0.0
        self.final_internal_coordinate_norm = 0.0
        self.max_internal_vertex_displacement = 0.0
        self.final_internal_vertex_displacement = 0.0
        self.max_modal_update = 0.0
        self.final_modal_residual_ratio = 0.0
        self.min_frame_z = self.bottom_z
        self.max_frame_z = self.bottom_z
        self.bottom_pause_steps = 0
        self.top_pause_steps = 0
        self.max_interface_gap = 0.0
        self.max_interface_angle_error = 0.0
        self.bottom_min_angle = 0.0
        self.bottom_max_angle = 0.0
        self.top_min_angle = 0.0
        self.top_max_angle = 0.0
        self.final_platform_angle = 0.0

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=32,
            rigid_joint_linear_ke=1.0e12,
            rigid_joint_angular_ke=1.0e12,
            rigid_joint_adaptive_stiffness=False,
        )
        self._apply_parameter_scales()

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 2.0e-5
        bounds_min = np.array([-0.75, -0.3, self.bottom_z - 0.25])
        bounds_max = np.array([0.4, 0.3, self.top_z + 0.25])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.45, -1.0, 0.25]))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _frame_z(self, time: float) -> float:
        phase = time % self.motion_period
        if phase < self.endpoint_duration:
            return self.bottom_z

        phase -= self.endpoint_duration
        if phase < self.travel_duration:
            fraction = self._smoothstep(phase / self.travel_duration)
            return self.bottom_z + self.stroke * fraction

        phase -= self.travel_duration
        if phase < self.endpoint_duration:
            return self.top_z

        phase -= self.endpoint_duration
        fraction = self._smoothstep(phase / self.travel_duration)
        return self.top_z - self.stroke * fraction

    def _endpoint_angle(self, time: float) -> float:
        duration = self.rotation_leg_duration
        if time < duration:
            return self.rotation_amplitude * self._smoothstep(time / duration)

        time -= duration
        if time < duration:
            return self.rotation_amplitude * (1.0 - 2.0 * self._smoothstep(time / duration))

        time -= duration
        if time < duration:
            return -self.rotation_amplitude * (1.0 - self._smoothstep(time / duration))

        return 0.0

    def _platform_angle(self, time: float) -> float:
        phase = time % self.motion_period
        if phase < self.endpoint_duration:
            return self._endpoint_angle(phase)

        phase -= self.endpoint_duration + self.travel_duration
        if 0.0 <= phase < self.endpoint_duration:
            return self._endpoint_angle(phase)

        return 0.0

    def _platform_orientation(self, time: float) -> wp.quat:
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(self._platform_angle(time)))

    def _drive_targets(self, time: float) -> dict[int, tuple[wp.vec3, wp.quat]]:
        return {self.base: (wp.vec3(-0.5, 0.0, self._frame_z(time)), self._platform_orientation(time))}

    def _elastic_frame_targets(self, time: float) -> dict[int, tuple[wp.vec3, wp.quat]]:
        orientation = self._platform_orientation(time)
        interface_world = wp.vec3(-0.5, 0.0, self._frame_z(time))
        interface_local = wp.vec3(-0.5, 0.0, 0.0)
        frame_position = interface_world - wp.quat_rotate(orientation, interface_local)
        return {self.body: (frame_position, orientation)}

    def _target_velocities(
        self,
        targets: dict[int, tuple[wp.vec3, wp.quat]],
        previous_targets: dict[int, tuple[wp.vec3, wp.quat]],
        time: float,
        previous_time: float,
    ) -> dict[int, tuple[wp.vec3, wp.vec3]]:
        velocities = finite_difference_target_velocities(targets, previous_targets, self.sim_dt)
        angle_rate = (self._platform_angle(time) - self._platform_angle(previous_time)) / self.sim_dt
        angular_velocity = wp.vec3(angle_rate, 0.0, 0.0)
        return {body: (linear, angular_velocity) for body, (linear, _angular) in velocities.items()}

    def _apply_parameter_scales(self) -> None:
        self.model.elastic_mode_mass.assign(self._base_mode_mass * self.mass_scale)
        self.model.elastic_mode_stiffness.assign(self._base_mode_stiffness * self.stiffness_scale)
        self.model.elastic_mode_damping.assign(self._base_mode_damping * self.damping_scale)
        self.model.elastic_mode_coupling_linear.assign(self._base_coupling_linear * self.mass_scale)
        self.model.elastic_mode_coupling_angular.assign(self._base_coupling_angular * self.mass_scale)
        self.model.elastic_mode_coupling_centrifugal.assign(self._base_coupling_centrifugal * self.mass_scale)
        self.model.elastic_mode_coupling_coriolis.assign(self._base_coupling_coriolis * self.mass_scale)

        body_mass = self._base_body_mass.copy()
        body_inv_mass = self._base_body_inv_mass.copy()
        body_inertia = self._base_body_inertia.copy()
        body_inv_inertia = self._base_body_inv_inertia.copy()
        body_mass[self.body] *= self.mass_scale
        body_inv_mass[self.body] /= self.mass_scale
        body_inertia[self.body] *= self.mass_scale
        body_inv_inertia[self.body] /= self.mass_scale
        self.model.body_mass.assign(body_mass)
        self.model.body_inv_mass.assign(body_inv_mass)
        self.model.body_inertia.assign(body_inertia)
        self.model.body_inv_inertia.assign(body_inv_inertia)
        self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def _vertex_displacement(self, q: np.ndarray) -> float:
        displacement = np.einsum(
            "smc,m->sc",
            self.basis.sample_phi[: self.surface_sample_count],
            q,
        )
        return float(np.max(np.linalg.norm(displacement, axis=1)))

    def _internal_mode_metrics(self, q: np.ndarray) -> tuple[float, float]:
        """Return fixed-interface coordinate norm and recovered displacement."""
        reduced_coordinates = self.generator.modal_matrix @ q
        internal_coordinates = reduced_coordinates[self.generator.interface_dof_count :]
        internal_displacement = (self._internal_recovery @ internal_coordinates).reshape((-1, 3))
        return (
            float(np.linalg.norm(internal_coordinates)),
            float(np.max(np.linalg.norm(internal_displacement, axis=1))),
        )

    def _update_metrics(self, frame_z: float, frame_speed: float, platform_angle: float) -> None:
        modes = self.state_0.joint_q.numpy()[self.modal_q_slice]
        self.final_mode_norm = float(np.linalg.norm(modes))
        self.max_mode_norm = max(self.max_mode_norm, self.final_mode_norm)
        self.final_vertex_displacement = self._vertex_displacement(modes)
        self.max_vertex_displacement = max(self.max_vertex_displacement, self.final_vertex_displacement)
        (
            self.final_internal_coordinate_norm,
            self.final_internal_vertex_displacement,
        ) = self._internal_mode_metrics(modes)
        self.max_internal_coordinate_norm = max(
            self.max_internal_coordinate_norm,
            self.final_internal_coordinate_norm,
        )
        self.max_internal_vertex_displacement = max(
            self.max_internal_vertex_displacement,
            self.final_internal_vertex_displacement,
        )
        self.min_frame_z = min(self.min_frame_z, frame_z)
        self.max_frame_z = max(self.max_frame_z, frame_z)
        body_q = self.state_0.body_q.numpy()
        fixture_interface = body_q[self.base, :3]
        elastic_interface = body_q[self.body, :3] + quat_rotate(body_q[self.body, 3:7], np.array([-0.5, 0.0, 0.0]))
        self.max_interface_gap = max(
            self.max_interface_gap,
            float(np.linalg.norm(elastic_interface - fixture_interface)),
        )
        base_orientation = body_q[self.base, 3:7]
        elastic_orientation = body_q[self.body, 3:7]
        orientation_dot = abs(
            float(np.dot(base_orientation, elastic_orientation))
            / max(float(np.linalg.norm(base_orientation) * np.linalg.norm(elastic_orientation)), 1.0e-12)
        )
        angle_error = 2.0 * np.arccos(np.clip(orientation_dot, 0.0, 1.0))
        self.max_interface_angle_error = max(self.max_interface_angle_error, float(angle_error))
        self.final_platform_angle = platform_angle
        if abs(frame_speed) < 1.0e-5:
            if abs(frame_z - self.bottom_z) < 1.0e-5:
                self.bottom_pause_steps += 1
                self.bottom_min_angle = min(self.bottom_min_angle, platform_angle)
                self.bottom_max_angle = max(self.bottom_max_angle, platform_angle)
            elif abs(frame_z - self.top_z) < 1.0e-5:
                self.top_pause_steps += 1
                self.top_min_angle = min(self.top_min_angle, platform_angle)
                self.top_max_angle = max(self.top_max_angle, platform_angle)

        metrics = self.solver.elastic_mode_solve_metrics()
        initial_residual = float(metrics["initial_residual_norm"][0])
        solve_residual = float(metrics["solve_residual_norm"][0])
        self.final_modal_residual_ratio = solve_residual / max(initial_residual, 1.0e-12)
        self.max_modal_update = max(self.max_modal_update, float(metrics["update_max"][0]))

    def step(self):
        for substep in range(self.sim_substeps):
            motion_time = self.render_time + substep * self.frame_dt / self.sim_substeps
            previous_motion_time = max(motion_time - self.frame_dt / self.sim_substeps, 0.0)
            targets = self._drive_targets(motion_time)
            previous_targets = self._drive_targets(previous_motion_time)
            velocities = self._target_velocities(targets, previous_targets, motion_time, previous_motion_time)
            apply_kinematic_targets(
                self.state_0,
                self._body_q_starts,
                targets,
                velocities,
                self._body_qd_starts,
            )
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            elastic_targets = self._elastic_frame_targets(motion_time)
            previous_elastic_targets = self._elastic_frame_targets(previous_motion_time)
            elastic_velocities = self._target_velocities(
                elastic_targets,
                previous_elastic_targets,
                motion_time,
                previous_motion_time,
            )
            # The fixed joint supplies the modal reaction during the solve. Project
            # the floating frame afterward so its reference interface is exact at
            # every rendered state instead of retaining finite penalty error.
            apply_kinematic_targets(
                self.state_0,
                self._elastic_frame_q_starts,
                elastic_targets,
                elastic_velocities,
                self._elastic_frame_qd_starts,
            )
            self.sim_time += self.sim_dt
            frame_z = self._frame_z(motion_time)
            frame_speed = float(velocities[self.base][0][2])
            self._update_metrics(frame_z, frame_speed, self._platform_angle(motion_time))
        self.render_time += self.frame_dt
        self.step_count += 1

    def render(self):
        self.viewer.begin_frame(self.render_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        changed = False
        mass_changed, self.mass_scale = ui.slider_float("Mass scale", self.mass_scale, 0.1, 5.0, "%.2f")
        stiffness_changed, self.stiffness_scale = ui.slider_float(
            "Stiffness scale", self.stiffness_scale, 0.1, 5.0, "%.2f"
        )
        damping_changed, self.damping_scale = ui.slider_float("Damping scale", self.damping_scale, 0.0, 10.0, "%.2f")
        changed = mass_changed or stiffness_changed or damping_changed

        if ui.button("Reset material scales"):
            self.mass_scale = self.default_mass_scale
            self.stiffness_scale = 1.0
            self.damping_scale = 1.0
            changed = True

        if changed:
            self._apply_parameter_scales()

    def test_final(self):
        if self.generator.fixed_interface_mode_count != 2:
            raise AssertionError(
                "expected two fixed-interface modes in the Craig-Bampton export, "
                f"got {self.generator.fixed_interface_mode_count}"
            )
        if self.basis.mode_count != 8:
            raise AssertionError(f"expected 8 retained elastic modes, got {self.basis.mode_count}")
        if self.generator.discarded_mode_count != 0:
            raise AssertionError(
                f"expected all Craig-Bampton coordinates to be retained, got "
                f"{self.generator.discarded_mode_count} discarded modes"
            )
        if np.linalg.norm(self.generator.modal_matrix[self.generator.interface_dof_count :]) <= 1.0e-8:
            raise AssertionError("retained modes do not use the fixed-interface input coordinates")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_frame_z - self.min_frame_z < 0.99 * self.stroke:
            raise AssertionError(
                f"reference interface did not traverse the requested stroke: {self.max_frame_z - self.min_frame_z} m"
            )
        if self.bottom_pause_steps < 10 or self.top_pause_steps < 10:
            raise AssertionError(
                f"reference interface did not pause at both ends: bottom={self.bottom_pause_steps}, "
                f"top={self.top_pause_steps}"
            )
        if self.max_interface_gap > 1.0e-6:
            raise AssertionError(f"left reference interface slipped by {self.max_interface_gap} m")
        if self.max_interface_angle_error > 1.0e-6:
            raise AssertionError(
                f"left reference interface rotated out of alignment by {self.max_interface_angle_error} rad"
            )
        if not np.allclose(
            self.model.elastic_mode_mass.numpy(),
            self._base_mode_mass * self.default_mass_scale,
        ):
            raise AssertionError("default 3x elastic mass scale was not applied")
        rotation_tolerance = np.deg2rad(1.0)
        target_angle_degrees = np.rad2deg(self.rotation_amplitude)
        for endpoint, min_angle, max_angle in (
            ("bottom", self.bottom_min_angle, self.bottom_max_angle),
            ("top", self.top_min_angle, self.top_max_angle),
        ):
            if min_angle > -self.rotation_amplitude + rotation_tolerance:
                raise AssertionError(
                    f"{endpoint} rotation did not reach -{target_angle_degrees:g} degrees: {np.rad2deg(min_angle)}"
                )
            if max_angle < self.rotation_amplitude - rotation_tolerance:
                raise AssertionError(
                    f"{endpoint} rotation did not reach +{target_angle_degrees:g} degrees: {np.rad2deg(max_angle)}"
                )
        if abs(self.final_platform_angle) > np.deg2rad(0.1):
            raise AssertionError(
                f"platform did not return to zero rotation: {np.rad2deg(self.final_platform_angle)} degrees"
            )
        if self.max_vertex_displacement < 1.0e-6:
            raise AssertionError(
                f"reference motion did not excite the Craig-Bampton modes: {self.max_vertex_displacement} m"
            )
        if self.max_internal_vertex_displacement < 1.0e-6:
            raise AssertionError(
                f"reference motion did not excite the fixed-interface modes: {self.max_internal_vertex_displacement} m"
            )
        internal_contribution = self.max_internal_vertex_displacement / self.max_vertex_displacement
        if internal_contribution < 0.1:
            raise AssertionError(
                "fixed-interface modes did not contribute materially to the recovered motion: "
                f"{internal_contribution:.3g}"
            )
        if self.final_vertex_displacement > 0.1 * self.max_vertex_displacement:
            raise AssertionError(
                "Craig-Bampton deformation did not settle during the final bottom pause: "
                f"peak={self.max_vertex_displacement}, final={self.final_vertex_displacement}"
            )
        if self.final_modal_residual_ratio > 5.0e-4:
            raise AssertionError(f"Craig-Bampton modal solve residual is too large: {self.final_modal_residual_ratio}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=540)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
