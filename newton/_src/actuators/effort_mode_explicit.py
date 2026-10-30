# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit effort mode: control law at the current state, then clamps."""

from __future__ import annotations

import inspect
from typing import Any

import warp as wp

from .clamping.base import ClampingBase
from .context import ActuatorContext
from .drives.base import DriveBase


class _EffortModeExplicit:
    """Explicit effort mode: control law at the current state, then clamps."""

    def __init__(
        self,
        drive: DriveBase,
        clamping: list[ClampingBase],
        device: wp.Device,
        *,
        context: ActuatorContext | None = None,
    ):
        self._drive = drive
        self._clamping = clamping
        self._device = device
        # Select each component's input keyword at construction, not every step.
        # Pass context only when explicitly declared: an older override may
        # accept **kwargs but ignore context, losing its supply-voltage input.
        self._drive_context_kwargs = self._context_kwargs(drive.compute, context)
        self._clamp_context_kwargs = [self._context_kwargs(c.modify_forces, context) for c in clamping]

    @staticmethod
    def _context_kwargs(method, context: ActuatorContext | None) -> dict[str, Any]:
        if context is None:
            return {}
        if "context" in inspect.signature(method).parameters:
            return {"context": context}
        return {"supply_voltage": context.supply_voltage}

    def is_graphable(self) -> bool:
        return self._drive.is_graphable()

    def compute_force(
        self,
        sim_state: Any,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        computed_forces: wp.array[float],
        applied_forces: wp.array[float] | None,
        drive_state: DriveBase.State | None,
        dt: float | None,
        custom_inputs: dict[str, Any] | None = None,
        bias_force: wp.array[float] | None = None,
        context: ActuatorContext | None = None,
    ) -> wp.array[float]:
        """Compute raw effort into *computed_forces*, clamp into *applied_forces*.

        Returns the buffer holding the final (clamped) effort.
        *bias_force* is ignored.
        """
        compute_kwargs: dict[str, Any] = {"device": self._device}
        if self._drive.custom_inputs:
            compute_kwargs["custom_inputs"] = custom_inputs
        compute_kwargs.update(self._drive_context_kwargs)
        self._drive.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            feedforward,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_vel_indices,
            computed_forces,
            drive_state,
            dt,
            **compute_kwargs,
        )
        forces = computed_forces
        for clamp, physical_inputs in zip(self._clamping, self._clamp_context_kwargs, strict=True):
            clamp.modify_forces(
                forces,
                applied_forces,
                positions,
                velocities,
                pos_indices,
                vel_indices,
                device=self._device,
                **physical_inputs,
            )
            forces = applied_forces
        return forces
