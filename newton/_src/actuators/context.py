# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical inputs shared by an actuator's components during evaluation."""

from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class ActuatorContext:
    """Shared physical inputs for this step; each component owns its internal state.

    The power stage updates these arrays before the drive and clamps run.
    Components must read them without modifying them. The arrays are reused
    between steps, so retaining the context does not preserve earlier values.
    """

    supply_voltage: wp.array[float] | None = None
    """Actual terminal voltage [V], shape (N,), or None without a power stage."""
