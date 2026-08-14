# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Index tables derived from a :class:`~newton.Model` and a joint selection.

:class:`_DofIndices` is a lookup table, not a concept: everything in it is
recomputable from ``(model, selection)``, and it exists so the derivation
happens once, lands on the device, and can be tested without constructing a
controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.model import Model

# Joints whose position error ``q_des - q`` is a well-defined scalar subtraction.
# A quaternion difference is not, so multi-DOF joints cannot be controlled.
_SCALAR_JOINT_TYPES = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))


@dataclass(frozen=True)
class _DofIndices:
    """Where the controlled DOFs live in the model's arrays.

    Positions and velocities are indexed differently in Newton -- a free joint
    spans 7 coordinates but 6 DOFs -- so a controlled DOF needs both a
    coordinate index (into :attr:`newton.State.joint_q`) and a DOF index (into
    :attr:`newton.State.joint_qd`).

    Robots are numbered by selection order, so robot *i* is
    ``selected_articulations[i]``. All arrays live on the model's device.
    """

    controlled_dof_to_model_dof: wp.array[wp.int32]
    """Model DOF index of each controlled DOF, shape [controlled_dof_count]."""

    controlled_dof_to_model_coord: wp.array[wp.int32]
    """Model coordinate index of each controlled DOF, shape [controlled_dof_count]."""

    dof_offsets: wp.array[wp.int32]
    """Start of each robot's controlled DOFs, plus sentinel, shape [robot_count + 1]."""

    selected_articulations: wp.array[wp.int32]
    """Model articulation index of each robot, shape [robot_count]."""

    articulation_dof_start: wp.array[wp.int32]
    """First model DOF of each robot's articulation, shape [robot_count]."""

    controlled_joints: tuple[int, ...]
    """Model joint index of each controlled DOF, in controlled order."""

    robot_count: int
    """Number of selected articulations."""

    controlled_dof_count: int
    """Total number of controlled DOFs."""

    max_dofs: int
    """Largest controlled DOF count of any single robot."""


def _resolve_articulations(model: Model, articulations: list[int] | list[str] | None) -> np.ndarray:
    """Resolve an articulation selection to model articulation indices.

    Args:
        model: Model the selection refers to.
        articulations: Articulation indices or labels. ``None`` selects all.

    Returns:
        Sorted, unique articulation indices.
    """
    if articulations is None:
        return np.arange(model.articulation_count, dtype=np.int64)

    if len(articulations) == 0:
        raise ValueError("articulations must not be empty; pass None to select all articulations.")

    resolved: list[int] = []

    for entry in articulations:
        if not isinstance(entry, str) and not isinstance(entry, int):
            raise ValueError(f"{entry} is not of type `str` or `int`")

        if isinstance(entry, str):
            matches = [i for i, label in enumerate(model.articulation_label) if label == entry]
            if not matches:
                raise ValueError(
                    f"articulation label {entry!r} matches no articulation in the model; "
                    f"available labels: {sorted(set(model.articulation_label))}."
                )
            resolved.extend(matches)
        else:
            index = int(entry)
            if not 0 <= index < model.articulation_count:
                raise ValueError(
                    f"articulation index {index} is out of range for a model with "
                    f"{model.articulation_count} articulations."
                )
            resolved.append(index)

    unique, counts = np.unique(np.asarray(resolved, dtype=np.int64), return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"articulations selects {unique[counts > 1].tolist()} more than once.")
    return unique


def _joint_to_articulation(model: Model, joints: np.ndarray) -> np.ndarray:
    """Map model joint indices to the articulation whose tree they belong to.

    Args:
        model: Model the joints belong to.
        joints: Model joint indices.

    Returns:
        Articulation index of each joint.

    Raises:
        ValueError: If a joint is outside every articulation, or is a
            loop-closing joint rather than a regular tree joint.
    """
    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()

    candidate = np.searchsorted(art_start, joints, side="right") - 1
    in_range = (candidate >= 0) & (candidate < model.articulation_count)
    # articulation_start bounds loop-closing joints too; articulation_end does not.
    is_tree_joint = in_range & (joints < art_end[np.clip(candidate, 0, model.articulation_count - 1)])
    if not np.all(is_tree_joint):
        bad = joints[~is_tree_joint].tolist()
        raise ValueError(
            f"joints {bad} are not regular tree joints of any articulation; "
            "only joints belonging to an articulation can be controlled."
        )
    return candidate


def _resolve_joints(
    model: Model,
    joints: list[int] | list[str] | None,
    selected_articulations: np.ndarray,
) -> np.ndarray:
    """Resolve a joint selection to model joint indices within the selected articulations.

    Integers match exactly; labels match every joint carrying that label.

    Args:
        model: Model the selection refers to.
        joints: Joint indices or labels. ``None`` selects every scalar joint of
            the selected articulations.
        selected_articulations: Articulation indices the selection is scoped to.

    Returns:
        Model joint indices in resolution order. Ordering is imposed later by
        :func:`_build_dof_indices`, which is the single place it is decided.
    """
    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()
    joint_type = model.joint_type.numpy()

    in_scope = np.concatenate([np.arange(art_start[a], art_end[a], dtype=np.int64) for a in selected_articulations])

    if joints is None:
        return in_scope[np.isin(joint_type[in_scope], _SCALAR_JOINT_TYPES)]

    if len(joints) == 0:
        raise ValueError("joints must not be empty; pass None to select all scalar joints.")

    labels = model.joint_label
    in_scope_set = set(in_scope.tolist())
    resolved: list[int] = []
    for entry in joints:
        if isinstance(entry, str):
            matches = [j for j in in_scope.tolist() if labels[j] == entry]
            if not matches:
                # Distinguish "no such label" from "label names a joint we cannot control",
                # which is what a loop-closing joint looks like from here.
                elsewhere = [j for j, label in enumerate(labels) if label == entry]
                if elsewhere:
                    raise ValueError(
                        f"joint label {entry!r} matches joints {elsewhere} in the model, but none "
                        "are regular tree joints of the selected articulations; loop-closing "
                        "joints and joints outside the selection cannot be controlled."
                    )
                raise ValueError(
                    f"joint label {entry!r} matches no joint in the selected articulations; "
                    f"available labels: {sorted({labels[j] for j in in_scope.tolist()})}."
                )
            resolved.extend(matches)
        else:
            index = int(entry)
            if index not in in_scope_set:
                raise ValueError(f"joint index {index} is not a regular tree joint of the selected articulations.")
            resolved.append(index)

    selection = np.asarray(resolved, dtype=np.int64)
    unique, counts = np.unique(selection, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"joints selects {unique[counts > 1].tolist()} more than once.")
    return selection


def select_joints(
    model: Model,
    *,
    articulations: list[int] | list[str] | None = None,
    exclude_articulations: list[int] | list[str] | None = None,
    joints: list[int] | list[str] | None = None,
    exclude_joints: list[int] | list[str] | None = None,
) -> list[int]:
    """Resolve an articulation and joint selection to model joint indices.

    Selecting whole articulations is a one-liner, but two common shapes are not
    expressible that way: a robot that is only partly controlled, because
    ``joints`` applies within *every* selected articulation, and a fleet where
    one kind is to be left out, because listing the kinds you keep scales with
    the fleet rather than with the exclusion. The two ``exclude_*`` arguments
    close both gaps.

    Exclusions are applied after the corresponding inclusion, and articulations
    are resolved before joints, so ``exclude_joints`` sees only the articulations
    that survived ``exclude_articulations``.

    The result is ordered by model joint index and is intended for the
    ``joints`` argument of :class:`~newton.controllers.ControllerJointImpedance`
    or :class:`~newton.controllers.ControllerDifferentialKinematics`.

    Selection follows the controllers' own rules: integers match exactly, labels
    match every entry carrying them, and a selector matching nothing is an error
    rather than an empty selection. A label is matched by exact string equality
    and always takes everything that carries it -- if two robot kinds share a
    joint label, both are selected.

    Args:
        model: Model to select from.
        articulations: Articulation indices or labels to control. ``None``
            selects all.
        exclude_articulations: Articulation indices or labels to drop. One label
            covers every replica of a robot kind, so leaving one kind out of a
            ten-kind scene costs one string rather than nine.
        joints: Joint indices or labels within the surviving articulations.
            ``None`` selects every scalar joint.
        exclude_joints: Joint indices or labels to drop from the result. One
            label covers a whole replicated fleet, so excluding ``"gripper"``
            scales with the number of joint *types*, not the number of robots.

    Returns:
        Model joint indices, sorted ascending.

    Raises:
        ValueError: If any selector matches nothing, if an exclusion removes
            nothing from the selection, or if it removes everything.

    Example:
        .. code-block:: python

            # every robot except one kind, minus each survivor's gripper
            joints = select_joints(model, exclude_articulations=["crane"], exclude_joints=["gripper"])
            controller = ControllerJointImpedance(model, joints=joints, ...)
    """
    selected_arts = _resolve_articulations(model, articulations)

    if exclude_articulations is not None:
        dropped = _resolve_articulations(model, exclude_articulations)
        remaining = np.setdiff1d(selected_arts, dropped)
        if remaining.size == selected_arts.size:
            raise ValueError(
                f"exclude_articulations={exclude_articulations!r} removed no articulation from the "
                f"selection; it resolved to {sorted(dropped.tolist())}, none of which were selected."
            )
        if remaining.size == 0:
            raise ValueError(f"exclude_articulations={exclude_articulations!r} removed every selected articulation.")
        selected_arts = remaining

    selected = _resolve_joints(model, joints, selected_arts)

    if exclude_joints is not None:
        dropped = _resolve_joints(model, exclude_joints, selected_arts)
        remaining = np.setdiff1d(selected, dropped)
        if remaining.size == selected.size:
            raise ValueError(
                f"exclude_joints={exclude_joints!r} removed no joint from the selection; it resolved "
                f"to {sorted(dropped.tolist())}, none of which were selected."
            )
        if remaining.size == 0:
            raise ValueError(f"exclude_joints={exclude_joints!r} removed every selected joint.")
        selected = remaining

    return sorted(selected.tolist())


def _build_dof_indices(
    model: Model,
    *,
    articulations: list[int] | list[str] | None = None,
    joints: list[int] | list[str] | None = None,
) -> _DofIndices:
    """Expand a joint selection into the controller's index tables.

    Pure host-side derivation; the only device work is uploading the results.
    Controlled DOFs are ordered by articulation, then by model joint index --
    *not* by the order they were listed -- so per-DOF gains are laid out
    robot-major.

    Args:
        model: Model to select from.
        articulations: Articulation indices or labels to control. ``None``
            selects every articulation.
        joints: Joint indices or labels to control, resolved within the
            selected articulations. ``None`` selects every scalar joint.

    Returns:
        Index tables addressing the selected DOFs.

    Raises:
        ValueError: If the model has no articulations, if a selector matches
            nothing, if a label is ambiguous, if a selection repeats an entry,
            or if a selected joint is not 1-DOF revolute or prismatic.
    """
    if model.articulation_count == 0:
        raise ValueError("model contains no articulations; nothing can be controlled.")

    selected_arts = _resolve_articulations(model, articulations)
    selected_joints = _resolve_joints(model, joints, selected_arts)

    joint_type = model.joint_type.numpy()
    art_start_all = model.articulation_start.numpy()
    art_end_all = model.articulation_end.numpy()
    unsupported = [
        (int(j), JointType(joint_type[j]).name) for j in selected_joints if joint_type[j] not in _SCALAR_JOINT_TYPES
    ]
    if unsupported:
        raise ValueError(
            f"controlled joints must be 1-DOF revolute or prismatic; got {unsupported}. "
            "Uncontrolled joints may be of any type."
        )
    if selected_joints.size == 0:
        # joints=None silently skips non-scalar joints, which is what lets a
        # floating base be read but not actuated. Say so when nothing is left.
        skipped = [
            (int(j), JointType(joint_type[j]).name)
            for a in selected_arts
            for j in range(art_start_all[a], art_end_all[a])
            if joint_type[j] not in _SCALAR_JOINT_TYPES
        ]
        raise ValueError(
            "selection resolved to zero controlled joints; controlled joints must be "
            f"1-DOF revolute or prismatic, and the selected articulations contain only {skipped}."
        )

    owning_art = _joint_to_articulation(model, selected_joints)
    order = np.lexsort((selected_joints, owning_art))
    selected_joints, owning_art = selected_joints[order], owning_art[order]

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    art_start = model.articulation_start.numpy()

    robots, dofs_per_robot = np.unique(owning_art, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(dofs_per_robot)]).astype(np.int32)

    device = model.device
    return _DofIndices(
        controlled_dof_to_model_dof=wp.array(qd_start[selected_joints], dtype=wp.int32, device=device),
        controlled_dof_to_model_coord=wp.array(q_start[selected_joints], dtype=wp.int32, device=device),
        dof_offsets=wp.array(offsets, dtype=wp.int32, device=device),
        selected_articulations=wp.array(robots, dtype=wp.int32, device=device),
        articulation_dof_start=wp.array(qd_start[art_start[robots]], dtype=wp.int32, device=device),
        controlled_joints=tuple(int(j) for j in selected_joints),
        robot_count=int(robots.size),
        controlled_dof_count=int(selected_joints.size),
        max_dofs=int(dofs_per_robot.max()),
    )
