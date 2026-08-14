# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for :mod:`newton.controllers`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from newton._src.sim.model import Model


def _validate_with_exact_shape(
    *,
    array: Any,
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
    device: wp.DeviceLike,
    required: bool = True,
) -> None:
    """Validate a ``wp.array``, requiring every axis to match ``shape`` exactly.

    Use for arrays the controller sizes itself -- gains, targets, index arrays,
    per-robot blocks. A mismatch means the caller misjudged how much is being
    controlled, so an over-long array is an error rather than a convenience.

    Args:
        array: Value to validate, or ``None`` for an omitted optional argument.
        name: Argument name, used in error messages.
        dtype: Warp dtype the array must have.
        shape: Shape the array must have. A ``-1`` entry accepts any size in
            that axis, so ``(-1,)`` means "1-D, any length".
        device: Device the array must live on.
        required: Whether ``None`` is rejected.

    Raises:
        TypeError: If the value is not a ``wp.array``, or has the wrong dtype.
        ValueError: If it is missing while required, or is on the wrong device,
            or any axis differs from ``shape``.
    """
    if array is None:
        if required:
            raise ValueError(f"{name} is required, cannot be `None`.")
        return
    if not isinstance(array, wp.array):
        raise TypeError(f"{name} must be a wp.array, got {type(array).__name__}.")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}.")
    if array.device != device:
        raise ValueError(f"{name} must be on device {device}, got {array.device}.")
    actual = tuple(array.shape)
    if len(actual) != len(shape) or any(want not in (-1, got) for got, want in zip(actual, shape, strict=True)):
        expected = "(" + ", ".join("*" if d == -1 else str(d) for d in shape) + ")"
        raise ValueError(f"{name} must have shape {expected}, got {actual}.")


def _validate_with_minimum_shape(
    *,
    array: Any,
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
    device: wp.DeviceLike,
    required: bool = True,
) -> None:
    """Validate a ``wp.array``, requiring every axis to be at least ``shape``.

    Use for arrays the caller owns and the controller only indexes into --
    simulation state and control buffers. Their size is the caller's business;
    all the controller needs is that every index it will touch is in range, so
    binding a larger array is normal. The bound is therefore the highest index
    used, plus one.

    Identical to :func:`_validate_with_exact_shape` except that each axis is
    compared with ``>=`` rather than ``==``.

    Args:
        array: Value to validate, or ``None`` for an omitted optional argument.
        name: Argument name, used in error messages.
        dtype: Warp dtype the array must have.
        shape: Smallest shape the array may have. A ``-1`` entry accepts any
            size in that axis.
        device: Device the array must live on.
        required: Whether ``None`` is rejected.

    Raises:
        TypeError: If the value is not a ``wp.array``, or has the wrong dtype.
        ValueError: If it is missing while required, or is on the wrong device,
            or any axis is smaller than ``shape``.
    """
    if array is None:
        if required:
            raise ValueError(f"{name} is required, cannot be `None`.")
        return
    if not isinstance(array, wp.array):
        raise TypeError(f"{name} must be a wp.array, got {type(array).__name__}.")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}.")
    if array.device != device:
        raise ValueError(f"{name} must be on device {device}, got {array.device}.")
    actual = tuple(array.shape)
    if len(actual) != len(shape) or any(want != -1 and got < want for got, want in zip(actual, shape, strict=True)):
        expected = "(" + ", ".join("*" if d == -1 else str(d) for d in shape) + ")"
        raise ValueError(f"{name} must have shape at least {expected}, got {actual}.")


def _validate_articulations_are_world_rooted(model: Model, selected: np.ndarray) -> None:
    """Reject articulations mounted on a body no articulation drives.

    :func:`newton.eval_inverse_dynamics_passive` allocates its per-body scratch
    with ``wp.empty`` and relies on every body being written by an articulation
    traversal. A body outside the articulation being evaluated is never written,
    so an articulation attached to one reads recycled memory: the resulting
    gravity and Coriolis terms are wrong and differ between runs and between
    steps. Refusing is the only safe response, since nothing downstream can
    detect it.

    Args:
        model: Model the articulations belong to.
        selected: Indices of the articulations to check.

    Raises:
        ValueError: If a selected articulation has a joint whose parent body is
            driven by no joint of that same articulation.
    """
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()

    offenders = []
    for articulation in selected:
        joints = range(int(art_start[articulation]), int(art_end[articulation]))
        driven_here = {int(joint_child[j]) for j in joints}
        for j in joints:
            parent = int(joint_parent[j])
            if parent != -1 and parent not in driven_here:
                offenders.append((int(articulation), j, parent))

    if offenders:
        raise ValueError(
            "each controlled articulation must be rooted at the world or on a body it drives "
            f"itself; found (articulation, joint, external parent body) {offenders}. The joint "
            "attaching it to that body must belong to the same articulation -- add it to "
            "add_articulation() and leave it out of `joints` to read it without actuating it. "
            "Newton's inverse dynamics would otherwise read uninitialised state for that body "
            "and produce gravity and Coriolis terms that vary between runs."
        )
