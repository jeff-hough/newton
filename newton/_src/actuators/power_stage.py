# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select supply voltage and apply motor commands in the existing actuator pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

from .context import ActuatorContext

if TYPE_CHECKING:
    from .motor_dc import MotorDC


@wp.kernel
def _supply_voltage_kernel(
    vin: wp.array[float],
    supply_from_battery: wp.array[float],
    battery_voltage: wp.array[float],
    actuator_indices: wp.array[wp.uint32],
    supply_voltage: wp.array[float],
):
    i = wp.tid()
    voltage = vin[i]
    if supply_from_battery[i] != 0.0:
        voltage = battery_voltage[actuator_indices[i]]
    supply_voltage[i] = voltage


@wp.kernel
def _duty_to_voltage_kernel(duty: wp.array[float], supply_voltage: wp.array[float], voltage: wp.array[float]):
    i = wp.tid()
    voltage[i] = duty[i] * supply_voltage[i]


class _PowerStageDC:
    """Prepare supply inputs and convert final signed duty into motor voltage and effort."""

    def __init__(
        self,
        motor: MotorDC,
        actuator_indices: wp.array[wp.uint32] | None,
        *,
        voltage_attr: str,
        current_attr: str,
        device: wp.Device,
        count: int,
        requires_grad: bool,
    ):
        from .motor_dc import MotorDC  # noqa: PLC0415 - avoid a circular import with motor_dc

        self.motor = motor
        self._supply_override = (
            motor.compute_supply_voltage
            if type(motor).compute_supply_voltage is not MotorDC.compute_supply_voltage
            else None
        )
        self.actuator_indices = actuator_indices
        self.voltage_attr = voltage_attr
        self.current_attr = current_attr
        self.device = device
        self.context = ActuatorContext(
            supply_voltage=wp.zeros(count, dtype=float, device=device, requires_grad=requires_grad)
        )
        self._motor_voltage = wp.zeros(count, dtype=float, device=device, requires_grad=requires_grad)
        # Keep duty intact for the voltage conversion's backward pass.
        self._motor_effort = wp.zeros(count, dtype=float, device=device, requires_grad=requires_grad)
        self._supply_current = None

    def prepare(self, control: Any) -> ActuatorContext:
        """Update supply voltage and select the control array that will receive supply current."""
        from .actuator import _require_array  # noqa: PLC0415 - avoid a circular import with actuator

        battery_voltage = None
        self._supply_current = None
        if self.motor.uses_battery:
            battery_voltage = _require_array(control, self.voltage_attr)
            self._supply_current = _require_array(control, self.current_attr)
        if self._supply_override is not None:
            self._supply_override(
                battery_voltage, self.actuator_indices, self.context.supply_voltage, device=self.device
            )
            return self.context
        wp.launch(
            _supply_voltage_kernel,
            dim=len(self.context.supply_voltage),
            inputs=[self.motor.vin, self.motor.supply_from_battery, battery_voltage, self.actuator_indices],
            outputs=[self.context.supply_voltage],
            device=self.device,
        )
        return self.context

    def compute(
        self,
        command: wp.array[float],
        velocities: wp.array[float],
        vel_indices: wp.array[wp.uint32],
    ) -> wp.array[float]:
        """Apply actual supply voltage to duty, then compute motor effort and supply draw.

        The drive and clamps are responsible for duty limits. This conversion
        uses actual supply voltage regardless of the limiter's voltage estimate.
        """
        wp.launch(
            _duty_to_voltage_kernel,
            dim=len(command),
            inputs=[command, self.context.supply_voltage],
            outputs=[self._motor_voltage],
            device=self.device,
        )
        self.motor.compute(
            self._motor_voltage,
            self._motor_effort,
            velocities,
            vel_indices,
            self.context.supply_voltage,
            self.actuator_indices,
            self._supply_current,
            device=self.device,
        )
        return self._motor_effort
