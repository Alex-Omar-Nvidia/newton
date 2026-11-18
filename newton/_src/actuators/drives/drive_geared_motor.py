# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import warp as wp

from .base import DriveBase


@wp.kernel
def _current_to_torque_kernel(
    measured_current: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    k_t: wp.array[float],
    gear_ratio: wp.array[float],
    output: wp.array[float],
):
    i = wp.tid()
    output[i] = gear_ratio[i] * k_t[i] * measured_current[vel_indices[i]]


class DriveGearedMotor(DriveBase):
    """Estimate joint torque from measured motor current through an ideal gearbox.

    Effort law::

        torque = gear_ratio * k_t * measured_current

    ``gear_ratio`` is motor speed divided by joint speed. Positive current
    produces positive joint torque. Supply signed motor winding current [A],
    not DC bus current, through ``sim_control.measured_current``. The array
    has shape ``(joint_dof_count,)`` and uses velocity-DOF indexing.

    This stateless drive outputs effort [N·m] for rotational DOFs. It ignores
    position/velocity targets and feedforward. Gearbox losses, friction,
    electrical dynamics, and implicit actuation are not modeled.

    This is a runtime measurement adapter configured through Python; USD
    authoring/import is not applicable to this runtime-only adapter.
    """

    custom_inputs = ("measured_current",)

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Require finite, positive torque constants and gear ratios."""
        resolved = {}
        for name in ("k_t", "gear_ratio"):
            if name not in args:
                raise ValueError(f"DriveGearedMotor requires {name}")
            value = args[name]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
            resolved[name] = value
        return resolved

    def __init__(self, k_t: wp.array[float], gear_ratio: wp.array[float]):
        """Initialize the per-actuator conversion parameters.

        Args:
            k_t: Motor torque constants [N·m/A], shape ``(N,)``.
            gear_ratio: Positive motor-to-joint speed ratios, shape ``(N,)``.

        Raises:
            ValueError: Parameters are not matching one-dimensional float32
                arrays on the same device.
        """
        if k_t.ndim != 1 or gear_ratio.shape != k_t.shape:
            raise ValueError("k_t and gear_ratio must have matching one-dimensional shapes")
        if k_t.dtype != wp.float32 or gear_ratio.dtype != wp.float32:
            raise ValueError("k_t and gear_ratio must have dtype float32")
        if k_t.device != gear_ratio.device:
            raise ValueError("k_t and gear_ratio must be on the same device")
        self.k_t = k_t
        self.gear_ratio = gear_ratio

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Require one conversion parameter per actuator on the actuator device."""
        if len(self.k_t) != num_actuators:
            raise ValueError(f"k_t and gear_ratio must have length {num_actuators}")
        if self.k_t.device != device:
            raise ValueError("k_t and gear_ratio must be on the actuator device")

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        output: wp.array[float],
        state: DriveBase.State | None,
        dt: float,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
    ) -> None:
        """Write joint torque using the latest DOF-indexed measured current [A].

        Raises:
            ValueError: ``measured_current`` is missing or is not a float32
                Warp array matching the velocity shape and parameter device.
        """
        current = (custom_inputs or {}).get("measured_current")
        if current is None:
            raise ValueError("DriveGearedMotor requires sim_control.measured_current")
        if not isinstance(current, wp.array) or current.dtype != wp.float32:
            raise ValueError("measured_current must be a float32 Warp array")
        if current.ndim != 1 or current.shape != velocities.shape:
            raise ValueError("measured_current must be one-dimensional with one entry per joint DOF")
        if current.device != self.k_t.device:
            raise ValueError("measured_current must be on the same device as k_t and gear_ratio")
        wp.launch(
            kernel=_current_to_torque_kernel,
            dim=len(output),
            inputs=[current, vel_indices, self.k_t, self.gear_ratio],
            outputs=[output],
            device=device or self.k_t.device,
        )
