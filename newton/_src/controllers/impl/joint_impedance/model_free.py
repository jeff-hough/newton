# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedanceModelFree — joint-space impedance control with
caller-supplied dynamics terms.

Ports the caller owns (measurements, dynamics terms, torque output) are read
and written through optional index arrays, so they may be bound directly to
simulation-sized arrays. Ports the controller sizes (targets and gains) are
flat and indexed by controlled DOF, so index *k* is controlled DOF *k*.

The difference from :class:`ControllerJointImpedance` is that this controller
requires the caller to supply dynamics terms (mass matrix, gravity force,
Coriolis force) that the model-based controller computes internally.

Impedance law (terms enabled at construction):

    τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
        + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
        + [g(q)      if use_gravity_compensation  else 0]

where Δq = q_des - q and Δq̇ = q̇_des - q̇.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _validate_with_exact_shape, _validate_with_minimum_shape
from ._common import (
    _add_term_kernel,
    _gather_flat_kernel,
    _mass_matrix_multiply_kernel,
    _pd_term_kernel,
    _scatter_flat_kernel,
)


class ControllerJointImpedanceModelFree(ControllerBase):
    """Joint-space impedance controller with caller-supplied dynamics.

    Implements the joint-space impedance control law. This model-free variant
    expects the mass matrix, gravity, and Coriolis terms to be computed
    externally — it is the caller's responsibility to compute the enabled ones
    correctly and write them into the input struct before every :meth:`step`.

    Array shapes and devices are validated on each direct call to :meth:`step`,
    but not when a captured graph is replayed, since the checks run in Python
    at capture time only.

    Supports heterogeneous robot fleets — robots in the batch may have
    different DOF counts. Joint-space vectors are flat and ragged, so only the
    mass matrix is padded to ``max_dofs``.

    Allocate input and output structs via :meth:`input` and :meth:`output`.
    All field names on those structs are fixed — see :class:`Inputs` and
    :class:`Outputs` for the typed schema. Fields for disabled features
    (e.g. ``gravity_force`` when ``use_gravity_compensation=False``) are
    allocated as ``None`` and must not be written.

    See also :class:`ControllerJointImpedance`, which computes the mass matrix,
    gravity, and Coriolis terms internally from a Newton model.

    Args:
        dofs_per_robot: DOF count for each robot. Its length sets
            :attr:`robot_count`, its sum sets :attr:`controlled_dof_count`, and
            its maximum sets :attr:`max_dofs`, the padded width of the mass
            matrix. Robot *i* owns controlled DOFs
            ``[sum(dofs_per_robot[:i]), sum(dofs_per_robot[:i+1]))``.
        stiffness: Position-error gain Kp, shape ``(controlled_dof_count,)``.
            Units depend on ``use_inertia_decoupling``: [1/s²] when enabled,
            since the PD term is then an acceleration premultiplied by M(q);
            otherwise [N/m or N·m/rad]. Pass an array to copy it at
            construction, or ``None`` to read ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd, [1/s] when
            ``use_inertia_decoupling`` is enabled, otherwise
            [N·s/m or N·m·s/rad]. Same format as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
        joint_q_idx: Maps each controlled DOF to its slot in
            ``inputs.joint_q``, shape ``(controlled_dof_count,)``. Defaults to
            identity, meaning the port is already in controlled-DOF order.
        joint_qd_idx: Same, for ``inputs.joint_qd``. Kept separate from
            ``joint_q_idx`` because position and velocity need not share an
            index space — a free joint spans 7 coordinates but 6 DOFs.
        gravity_force_idx: Same, for ``inputs.gravity_force``.
        coriolis_force_idx: Same, for ``inputs.coriolis_force``.
        joint_f_idx: Maps each controlled DOF to the slot it writes in
            ``outputs.joint_f``. Must be free of duplicates.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedanceModelFree.input`.

        All four kinematic fields are always allocated. Optional fields are
        ``None`` when the corresponding feature is disabled at construction.
        """

        joint_q: wp.array[wp.float32]
        """Current joint positions [m or rad], read through ``joint_q_idx``."""
        joint_qd: wp.array[wp.float32]
        """Current joint velocities [m/s or rad/s], read through ``joint_qd_idx``."""
        joint_q_des: wp.array[wp.float32]
        """Desired joint positions [m or rad], shape ``(controlled_dof_count,)``."""
        joint_qd_des: wp.array[wp.float32]
        """Desired joint velocities [m/s or rad/s], shape ``(controlled_dof_count,)``."""
        joint_qdd: wp.array[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], shape ``(controlled_dof_count,)``. ``None`` unless ``has_qdd_feedforward=True``."""
        gravity_force: wp.array[wp.float32] | None
        """Gravity generalized forces [N or N·m], read through ``gravity_force_idx``. ``None`` unless ``use_gravity_compensation=True``."""
        coriolis_force: wp.array[wp.float32] | None
        """Coriolis generalized forces [N or N·m], read through ``coriolis_force_idx``. ``None`` unless ``use_coriolis_compensation=True``."""
        mass_matrix: wp.array3d[wp.float32] | None
        """Per-robot mass matrices, shape ``(robot_count, max_dofs, max_dofs)``. Units by row/column DOF type: [kg] translational, [kg·m] mixed, [kg·m²] rotational. ``None`` unless ``use_inertia_decoupling=True``."""
        stiffness: wp.array[wp.float32] | None
        """Position-error gain Kp, shape ``(controlled_dof_count,)``. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction."""
        damping: wp.array[wp.float32] | None
        """Velocity-error gain Kd, shape ``(controlled_dof_count,)``. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedanceModelFree.output`."""

        joint_f: wp.array[wp.float32]
        """Joint torque command [N or N·m], written through ``joint_f_idx``."""

    def __init__(
        self,
        *,
        dofs_per_robot: wp.array[wp.int32],
        stiffness: wp.array[wp.float32] | None,
        damping: wp.array[wp.float32] | None,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
        joint_q_idx: wp.array[wp.int32] | None = None,
        joint_qd_idx: wp.array[wp.int32] | None = None,
        gravity_force_idx: wp.array[wp.int32] | None = None,
        coriolis_force_idx: wp.array[wp.int32] | None = None,
        joint_f_idx: wp.array[wp.int32] | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        self._device = wp.get_device(device)

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere
        # else. dofs_per_robot comes first because the shapes below derive
        # from it.
        # ------------------------------------------------------------------
        _validate_with_exact_shape(
            array=dofs_per_robot, name="dofs_per_robot", dtype=wp.int32, shape=(-1,), device=self._device
        )

        dofs_per_robot_np = dofs_per_robot.numpy()
        robot_count = int(dofs_per_robot_np.size)
        if robot_count < 1:
            raise ValueError("dofs_per_robot must not be empty.")
        if dofs_per_robot_np.min() < 1:
            raise ValueError(f"every robot must have >= 1 DOF, got dofs_per_robot={dofs_per_robot_np.tolist()}.")

        max_dofs = int(dofs_per_robot_np.max())
        controlled_dof_count = int(dofs_per_robot_np.sum())
        flat_shape = (controlled_dof_count,)

        for name, array, dtype, required in (
            ("stiffness", stiffness, wp.float32, False),
            ("damping", damping, wp.float32, False),
            ("joint_q_idx", joint_q_idx, wp.int32, False),
            ("joint_qd_idx", joint_qd_idx, wp.int32, False),
            ("gravity_force_idx", gravity_force_idx, wp.int32, False),
            ("coriolis_force_idx", coriolis_force_idx, wp.int32, False),
            ("joint_f_idx", joint_f_idx, wp.int32, False),
        ):
            _validate_with_exact_shape(
                array=array, name=name, dtype=dtype, shape=flat_shape, device=self._device, required=required
            )
        # ------------------------------------------------------------------

        self._robot_count = robot_count
        self._max_dofs = max_dofs
        self._controlled_dof_count = controlled_dof_count
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._requires_grad = requires_grad

        self._dofs_per_robot = dofs_per_robot
        offsets_np = np.zeros(robot_count + 1, dtype=np.int32)
        offsets_np[1:] = np.cumsum(dofs_per_robot_np)
        self._dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=self._device)

        # Identity means "this port is already in controlled-DOF order".
        identity = wp.array(np.arange(controlled_dof_count, dtype=np.int32), dtype=wp.int32, device=self._device)
        self._q_idx = joint_q_idx if joint_q_idx is not None else identity
        self._qd_idx = joint_qd_idx if joint_qd_idx is not None else identity
        self._gravity_idx = gravity_force_idx if gravity_force_idx is not None else identity
        self._coriolis_idx = coriolis_force_idx if coriolis_force_idx is not None else identity
        self._f_idx = joint_f_idx if joint_f_idx is not None else identity

        f_idx_np = self._f_idx.numpy()
        if len(f_idx_np) != len(np.unique(f_idx_np)):
            raise ValueError(
                "joint_f output indices must be unique — two robots cannot scatter torques "
                "to the same simulation DOF slot."
            )

        # Smallest flat array each port may be bound to, so step() can reject a
        # short array before the gather/scatter kernels read out of bounds.
        self._min_len_q = int(np.max(self._q_idx.numpy())) + 1
        self._min_len_qd = int(np.max(self._qd_idx.numpy())) + 1
        self._min_len_gravity = int(np.max(self._gravity_idx.numpy())) + 1
        self._min_len_coriolis = int(np.max(self._coriolis_idx.numpy())) + 1
        self._min_len_f = int(np.max(self._f_idx.numpy())) + 1

        self._stiffness_baked = self._bake_gain(stiffness)
        self._damping_baked = self._bake_gain(damping)

        dev, rg = self._device, requires_grad
        self._q_flat = wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
        self._qd_flat = wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
        self._grav_flat: wp.array[wp.float32] | None = (
            wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
            if self._use_gravity
            else None
        )
        self._cor_flat: wp.array[wp.float32] | None = (
            wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
            if self._use_coriolis
            else None
        )

        self._tau_buf = wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
        self._acc_buf: wp.array[wp.float32] | None = (
            wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
            if self._use_inertia
            else None
        )

    def _bake_gain(self, value: wp.array[wp.float32] | None) -> wp.array[wp.float32] | None:
        """Copy a gain array so later edits to the caller's array have no effect.

        Returns ``None`` for live gains, which are read from the input struct
        each step instead. Already validated by :func:`_validate_with_exact_shape`.
        """
        if value is None:
            return None
        baked = wp.zeros(
            self._controlled_dof_count,
            dtype=wp.float32,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        wp.copy(baked, value)
        return baked

    @property
    def robot_count(self) -> int:
        return self._robot_count

    @property
    def max_dofs(self) -> int:
        return self._max_dofs

    @property
    def controlled_dof_count(self) -> int:
        return self._controlled_dof_count

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_graphable(self) -> bool:
        return True

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` with zero-initialised arrays."""
        d, rg = self._device, self._requires_grad
        n = self._controlled_dof_count

        inputs = ControllerJointImpedanceModelFree.Inputs()
        inputs.joint_q = wp.zeros(self._min_len_q, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd = wp.zeros(self._min_len_qd, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_q_des = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd_des = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qdd = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if self._has_qdd else None
        inputs.gravity_force = (
            wp.zeros(self._min_len_gravity, dtype=wp.float32, device=d, requires_grad=rg) if self._use_gravity else None
        )
        inputs.coriolis_force = (
            wp.zeros(self._min_len_coriolis, dtype=wp.float32, device=d, requires_grad=rg)
            if self._use_coriolis
            else None
        )
        inputs.mass_matrix = (
            wp.zeros((self._robot_count, self._max_dofs, self._max_dofs), dtype=wp.float32, device=d, requires_grad=rg)
            if self._use_inertia
            else None
        )
        inputs.stiffness = (
            wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if self._stiffness_baked is None else None
        )
        inputs.damping = (
            wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if self._damping_baked is None else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a flat torque array."""
        outputs = ControllerJointImpedanceModelFree.Outputs()
        outputs.joint_f = wp.zeros(
            self._min_len_f, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
        )
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Compute one impedance-control step and write joint torques.

        Args:
            inputs: Populated :class:`Inputs` struct. Dynamics fields must be
                filled by the caller before each call.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
        """
        stiffness = self._stiffness_baked if self._stiffness_baked is not None else inputs.stiffness
        damping = self._damping_baked if self._damping_baked is not None else inputs.damping

        n = self._controlled_dof_count
        flat_shape = (n,)

        # Indexed ports may be bound to larger caller arrays; controlled-order
        # ports must match the controlled DOF count exactly.
        _validate_with_minimum_shape(
            array=inputs.joint_q,
            name="inputs.joint_q",
            shape=(self._min_len_q,),
            device=self._device,
            dtype=wp.float32,
        )
        _validate_with_minimum_shape(
            array=inputs.joint_qd,
            name="inputs.joint_qd",
            shape=(self._min_len_qd,),
            device=self._device,
            dtype=wp.float32,
        )
        _validate_with_exact_shape(
            array=inputs.joint_q_des, name="inputs.joint_q_des", dtype=wp.float32, shape=flat_shape, device=self._device
        )
        _validate_with_exact_shape(
            array=inputs.joint_qd_des,
            name="inputs.joint_qd_des",
            dtype=wp.float32,
            shape=flat_shape,
            device=self._device,
        )
        if self._has_qdd:
            _validate_with_exact_shape(
                array=inputs.joint_qdd,
                name="inputs.joint_qdd",
                dtype=wp.float32,
                shape=flat_shape,
                device=self._device,
            )
        if self._use_gravity:
            _validate_with_minimum_shape(
                array=inputs.gravity_force,
                dtype=wp.float32,
                name="inputs.gravity_force",
                shape=(self._min_len_gravity,),
                device=self._device,
            )
        if self._use_coriolis:
            _validate_with_minimum_shape(
                array=inputs.coriolis_force,
                dtype=wp.float32,
                name="inputs.coriolis_force",
                shape=(self._min_len_coriolis,),
                device=self._device,
            )
        _validate_with_minimum_shape(
            array=outputs.joint_f,
            dtype=wp.float32,
            name="outputs.joint_f",
            shape=(self._min_len_f,),
            device=self._device,
        )

        _validate_with_exact_shape(
            array=stiffness, name="stiffness", dtype=wp.float32, shape=flat_shape, device=self._device
        )
        _validate_with_exact_shape(
            array=damping, name="damping", dtype=wp.float32, shape=flat_shape, device=self._device
        )
        if self._use_inertia:
            _validate_with_exact_shape(
                array=inputs.mass_matrix,
                name="inputs.mass_matrix",
                dtype=wp.float32,
                shape=(self._robot_count, self._max_dofs, self._max_dofs),
                device=self._device,
            )

        wp.launch(
            _gather_flat_kernel,
            dim=n,
            inputs=[inputs.joint_q, self._q_idx],
            outputs=[self._q_flat],
            device=self._device,
        )
        wp.launch(
            _gather_flat_kernel,
            dim=n,
            inputs=[inputs.joint_qd, self._qd_idx],
            outputs=[self._qd_flat],
            device=self._device,
        )
        if self._use_gravity:
            wp.launch(
                _gather_flat_kernel,
                dim=n,
                inputs=[inputs.gravity_force, self._gravity_idx],
                outputs=[self._grav_flat],
                device=self._device,
            )
        if self._use_coriolis:
            wp.launch(
                _gather_flat_kernel,
                dim=n,
                inputs=[inputs.coriolis_force, self._coriolis_idx],
                outputs=[self._cor_flat],
                device=self._device,
            )

        working_buf = self._acc_buf if self._use_inertia else self._tau_buf
        wp.launch(
            _pd_term_kernel,
            dim=n,
            inputs=[
                self._q_flat,
                self._qd_flat,
                inputs.joint_q_des,
                inputs.joint_qd_des,
                stiffness,
                damping,
            ],
            outputs=[working_buf],
            device=self._device,
        )

        if self._has_qdd:
            wp.launch(_add_term_kernel, dim=n, inputs=[inputs.joint_qdd], outputs=[working_buf], device=self._device)

        if self._use_inertia:
            wp.launch(
                _mass_matrix_multiply_kernel,
                dim=(self._robot_count, self._max_dofs),
                inputs=[inputs.mass_matrix, self._acc_buf, self._dof_offsets],
                outputs=[self._tau_buf],
                device=self._device,
            )

        if self._use_gravity:
            wp.launch(_add_term_kernel, dim=n, inputs=[self._grav_flat], outputs=[self._tau_buf], device=self._device)
        if self._use_coriolis:
            wp.launch(_add_term_kernel, dim=n, inputs=[self._cor_flat], outputs=[self._tau_buf], device=self._device)

        wp.launch(
            _scatter_flat_kernel,
            dim=n,
            inputs=[self._tau_buf, self._f_idx],
            outputs=[outputs.joint_f],
            device=self._device,
        )
