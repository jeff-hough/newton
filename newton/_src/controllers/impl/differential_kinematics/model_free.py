# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialKinematicsModelFree — differential IK with a
caller-supplied Jacobian.

Intended for frameworks that already compute a Jacobian through their own
physics engine. The controller skips forward kinematics and takes the Jacobian
and the current end-effector pose as input ports.

The Jacobian must use **Newton's COM-referenced convention** (linear rows give
the velocity of the link's centre of mass), matching :func:`newton.eval_jacobian`.
Converting from another convention is the caller's responsibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _validate_with_exact_shape, _validate_with_minimum_shape
from ._common import (
    _NON_CAPTURABLE_METHODS,
    DEFAULT_SOLVER_DAMPING,
    CommandType,
    IkMethod,
    _compact_qd_kernel,
    _gather_flat_kernel,
    _integrate_position_kernel,
    _null_space_correction_kernel,
    _quat_to_axis_angle,
    _scale_orient_error_rows_kernel,
    _scale_orient_jacobian_rows_kernel,
    _scatter_flat_kernel,
    run_solver,
)


@wp.kernel
def _compute_pose_error_kernel(
    ee_pos: wp.array[wp.vec3],
    ee_quat: wp.array[wp.quatf],
    target_pos: wp.array[wp.vec3],
    target_quat: wp.array[wp.quatf],
    e_buffer: wp.array2d[wp.float32],  # (N, 6)
):
    r = wp.tid()
    pos_err = target_pos[r] - ee_pos[r]
    e_buffer[r, 0] = pos_err[0]
    e_buffer[r, 1] = pos_err[1]
    e_buffer[r, 2] = pos_err[2]

    rot_err = _quat_to_axis_angle(target_quat[r] * wp.quat_inverse(ee_quat[r]))
    e_buffer[r, 3] = rot_err[0]
    e_buffer[r, 4] = rot_err[1]
    e_buffer[r, 5] = rot_err[2]


@wp.kernel
def _compute_position_error_kernel(
    ee_pos: wp.array[wp.vec3],
    target_pos: wp.array[wp.vec3],
    e_buffer: wp.array2d[wp.float32],  # (N, 6) — orientation rows zeroed for the padded solve
):
    r = wp.tid()
    pos_err = target_pos[r] - ee_pos[r]
    e_buffer[r, 0] = pos_err[0]
    e_buffer[r, 1] = pos_err[1]
    e_buffer[r, 2] = pos_err[2]
    e_buffer[r, 3] = float(0.0)
    e_buffer[r, 4] = float(0.0)
    e_buffer[r, 5] = float(0.0)


@wp.kernel
def _copy_jacobian_kernel(
    src: wp.array3d[wp.float32],
    dst: wp.array3d[wp.float32],
):
    r, i, j = wp.tid()
    dst[r, i, j] = src[r, i, j]


class ControllerDifferentialKinematicsModelFree(ControllerBase):
    """Differential IK from a caller-supplied Jacobian.

    Computes, for each robot, a joint-velocity command that drives the current
    end-effector pose toward a target, then integrates it to a joint-position
    target. Identical in error convention, solvers, orientation weighting, and
    null-space avoidance to :class:`ControllerDifferentialKinematics`, but
    without forward kinematics.

    Array shapes and devices are validated on each direct call to :meth:`step`,
    but not when a captured graph is replayed, since the checks run in Python
    at capture time only.

    Supports heterogeneous fleets — robots may have different DOF counts.
    Joint-space vectors are flat and ragged; only the Jacobian and the solver's
    scratch are padded to ``max_dofs``. A short robot's padding columns must be
    left at zero, which is what allows the solver to ignore them.

    Args:
        dofs_per_robot: DOF count for each robot. Its length sets
            :attr:`robot_count`, its sum sets :attr:`controlled_dof_count`, and
            its maximum sets :attr:`max_dofs`, the padded Jacobian width.
        bandwidth: Per-robot scale on the solved joint velocity, shape
            ``(robot_count,)``. Pass a float for a uniform value, an array to
            copy at construction, or ``None`` to read ``inputs.bandwidth`` each
            step.
        solver_damping: Per-robot damping λ for the damped least-squares solve,
            same format as ``bandwidth``. Ignored by TRANSPOSE, and forced to
            zero for PSEUDO_INVERSE.
        ik_method: Jacobian inverse strategy, see :class:`IkMethod`.
        command_type: Full pose or position only, see :class:`CommandType`.
        min_singular_value: Singular-value floor, SVD only.
        lambda_min: Baseline damping for adaptive damped least squares.
        lambda_max: Maximum damping for adaptive damped least squares.
        sigma_thresh: Singular-value threshold at which damping starts ramping.
        orientation_weight: Scalar or per-axis ``(wx, wy, wz)`` weight applied
            to the orientation rows of both the Jacobian and the error, letting
            an axis be de-emphasised or dropped without changing the task
            dimension. Requires ``CommandType.POSE``.
        joint_limit_avoidance_gain: Gain on the null-space centring bias. ``0``
            disables it.
        joint_limit_avoidance_margin: Distance from a limit [m or rad] at which
            the centring bias starts to act.
        joint_pos_lower: Lower joint limits [m or rad], shape
            ``(controlled_dof_count,)``. Required when avoidance is enabled.
        joint_pos_upper: Upper joint limits, same shape.
        joint_q_idx: Maps each controlled DOF to its slot in ``inputs.joint_q``.
            Defaults to identity, meaning the port is already in controlled-DOF
            order.
        joint_target_q_idx: Maps each controlled DOF to the slot it writes in
            ``outputs.joint_target_q``. Must be free of duplicates.
        joint_target_qd_idx: Same, for ``outputs.joint_target_qd``.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    IkMethod = IkMethod
    CommandType = CommandType
    DEFAULT_SOLVER_DAMPING: float = DEFAULT_SOLVER_DAMPING

    class Inputs:
        """Input struct returned by :meth:`~ControllerDifferentialKinematicsModelFree.input`."""

        joint_q: wp.array[wp.float32]
        """Current joint positions [m or rad], read through ``joint_q_idx``."""
        ee_pos: wp.array[wp.vec3]
        """Current end-effector position [m] in world frame, shape ``(robot_count,)``."""
        ee_quat: wp.array[wp.quatf]
        """Current end-effector orientation ``[x, y, z, w]`` in world frame, shape ``(robot_count,)``."""
        target_pos: wp.array[wp.vec3]
        """Desired end-effector position [m], shape ``(robot_count,)``."""
        target_quat: wp.array[wp.quatf] | None
        """Desired end-effector orientation ``[x, y, z, w]``, shape ``(robot_count,)``. ``None`` unless ``command_type`` is POSE."""
        jacobian: wp.array3d[wp.float32]
        """Geometric Jacobian in COM-referenced convention [m or rad per rad], shape ``(robot_count, task_dim, max_dofs)``. Padding columns of a short robot must be zero."""
        bandwidth: wp.array[wp.float32] | None
        """Per-robot velocity scale [dimensionless], shape ``(robot_count,)``. ``None`` when baked at construction."""
        solver_damping: wp.array[wp.float32] | None
        """Per-robot damped least-squares λ [dimensionless], shape ``(robot_count,)``. ``None`` when baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerDifferentialKinematicsModelFree.output`."""

        joint_target_q: wp.array[wp.float32]
        """Integrated joint position target [m or rad], written through ``joint_target_q_idx``."""
        joint_target_qd: wp.array[wp.float32]
        """Joint velocity command [m/s or rad/s], written through ``joint_target_qd_idx``."""

    def __init__(
        self,
        *,
        dofs_per_robot: wp.array[wp.int32],
        bandwidth: float | wp.array[wp.float32] | None = 1.0,
        solver_damping: float | wp.array[wp.float32] | None = DEFAULT_SOLVER_DAMPING,
        ik_method: IkMethod = IkMethod.DAMPED_LEAST_SQUARES,
        command_type: CommandType = CommandType.POSE,
        min_singular_value: float = 1e-5,
        lambda_min: float = 0.05,
        lambda_max: float = 0.2,
        sigma_thresh: float = 0.02,
        orientation_weight: float | tuple[float, float, float] | None = None,
        joint_limit_avoidance_gain: float = 0.0,
        joint_limit_avoidance_margin: float = 0.3,
        joint_pos_lower: wp.array[wp.float32] | None = None,
        joint_pos_upper: wp.array[wp.float32] | None = None,
        joint_q_idx: wp.array[wp.int32] | None = None,
        joint_target_q_idx: wp.array[wp.int32] | None = None,
        joint_target_qd_idx: wp.array[wp.int32] | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        self._device = wp.get_device(device)

        if not isinstance(ik_method, IkMethod):
            raise TypeError(f"ik_method must be IkMethod, got {type(ik_method).__name__}.")
        if not isinstance(command_type, CommandType):
            raise TypeError(f"command_type must be CommandType, got {type(command_type).__name__}.")
        if sigma_thresh <= 0.0:
            raise ValueError(f"sigma_thresh must be > 0, got {sigma_thresh}.")
        if lambda_min > lambda_max:
            raise ValueError(f"lambda_min ({lambda_min}) must be <= lambda_max ({lambda_max}).")
        if joint_limit_avoidance_gain < 0.0:
            raise ValueError(f"joint_limit_avoidance_gain must be >= 0, got {joint_limit_avoidance_gain}.")
        if joint_limit_avoidance_margin <= 0.0:
            raise ValueError(f"joint_limit_avoidance_margin must be > 0, got {joint_limit_avoidance_margin}.")
        if orientation_weight is not None and command_type != CommandType.POSE:
            raise ValueError(
                "orientation_weight requires CommandType.POSE; position-only control has no orientation rows."
            )

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

        limits_required = joint_limit_avoidance_gain > 0.0
        for name, array, dtype, shape, required in (
            ("joint_q_idx", joint_q_idx, wp.int32, flat_shape, False),
            ("joint_target_q_idx", joint_target_q_idx, wp.int32, flat_shape, False),
            ("joint_target_qd_idx", joint_target_qd_idx, wp.int32, flat_shape, False),
            ("joint_pos_lower", joint_pos_lower, wp.float32, flat_shape, limits_required),
            ("joint_pos_upper", joint_pos_upper, wp.float32, flat_shape, limits_required),
        ):
            _validate_with_exact_shape(
                array=array, name=name, dtype=dtype, shape=shape, device=self._device, required=required
            )
        # ------------------------------------------------------------------

        self._robot_count = robot_count
        self._max_dofs = max_dofs
        self._controlled_dof_count = controlled_dof_count
        self._command_type = command_type
        self._task_dim = int(command_type)
        self._ik_method = ik_method
        self._requires_grad = requires_grad
        self._min_singular_value = float(min_singular_value)
        self._lambda_min = float(lambda_min)
        self._lambda_max = float(lambda_max)
        self._sigma_thresh = float(sigma_thresh)
        self._avoidance_gain = float(joint_limit_avoidance_gain)
        self._avoidance_margin = float(joint_limit_avoidance_margin)
        self._joint_pos_lower = joint_pos_lower
        self._joint_pos_upper = joint_pos_upper

        offsets_np = np.zeros(robot_count + 1, dtype=np.int32)
        offsets_np[1:] = np.cumsum(dofs_per_robot_np)
        self._dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=self._device)

        identity = wp.array(np.arange(controlled_dof_count, dtype=np.int32), dtype=wp.int32, device=self._device)
        self._q_idx = joint_q_idx if joint_q_idx is not None else identity
        self._target_q_idx = joint_target_q_idx if joint_target_q_idx is not None else identity
        self._target_qd_idx = joint_target_qd_idx if joint_target_qd_idx is not None else identity

        for name, idx in (("joint_target_q_idx", self._target_q_idx), ("joint_target_qd_idx", self._target_qd_idx)):
            idx_np = idx.numpy()
            if len(idx_np) != len(np.unique(idx_np)):
                raise ValueError(f"{name} must be unique — two controlled DOFs cannot write the same output slot.")

        self._min_len_q = int(np.max(self._q_idx.numpy())) + 1
        self._min_len_target_q = int(np.max(self._target_q_idx.numpy())) + 1
        self._min_len_target_qd = int(np.max(self._target_qd_idx.numpy())) + 1

        self._orientation_weight = _normalize_orientation_weight(orientation_weight)

        self._bandwidth_baked = self._bake_gain(bandwidth, "bandwidth")
        self._damping_baked = self._bake_gain(solver_damping, "solver_damping")

        # Scratch. The 6-wide error and 6x6 system are fixed so one Cholesky
        # kernel serves both task dimensions.
        rg, dev = requires_grad, self._device
        self._e_buffer = wp.zeros((robot_count, 6), dtype=wp.float32, device=dev, requires_grad=rg)
        self._A = wp.zeros((robot_count, 6, 6), dtype=wp.float32, device=dev, requires_grad=rg)
        self._y = wp.zeros((robot_count, 6), dtype=wp.float32, device=dev, requires_grad=rg)
        self._qd_padded = wp.zeros((robot_count, max_dofs), dtype=wp.float32, device=dev, requires_grad=rg)
        self._qd_flat = wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
        self._q_flat = wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
        self._q_out_flat = wp.zeros(controlled_dof_count, dtype=wp.float32, device=dev, requires_grad=rg)
        self._zero_damping = wp.zeros(robot_count, dtype=wp.float32, device=dev)
        self._dt_buf = wp.zeros(1, dtype=wp.float32, device=dev)

        # Orientation weighting rewrites the Jacobian, so it needs a private copy.
        self._j_internal: wp.array3d[wp.float32] | None = None
        if self._orientation_weight is not None:
            self._j_internal = wp.zeros(
                (robot_count, self._task_dim, max_dofs), dtype=wp.float32, device=dev, requires_grad=rg
            )

        # Adaptive-DLS scratch, preallocated so no allocation happens per step.
        self._power_iter_v: wp.array2d[wp.float32] | None = None
        self._sigma_min_est: wp.array[wp.float32] | None = None
        self._adaptive_damping: wp.array[wp.float32] | None = None
        self._lambda_min_array: wp.array[wp.float32] | None = None
        if ik_method == IkMethod.ADAPTIVE_DAMPED_LEAST_SQUARES:
            self._power_iter_v = wp.zeros((robot_count, 6), dtype=wp.float32, device=dev)
            self._sigma_min_est = wp.zeros(robot_count, dtype=wp.float32, device=dev)
            self._adaptive_damping = wp.zeros(robot_count, dtype=wp.float32, device=dev)
            self._lambda_min_array = wp.full(robot_count, self._lambda_min, dtype=wp.float32, device=dev)

    def _bake_gain(self, value: float | wp.array[wp.float32] | None, name: str) -> wp.array[wp.float32] | None:
        """Copy a per-robot gain, or return ``None`` to read it live each step."""
        if value is None:
            return None
        if isinstance(value, wp.array):
            _validate_with_exact_shape(
                array=value, name=name, dtype=wp.float32, shape=(self._robot_count,), device=self._device
            )
            baked = wp.zeros(
                self._robot_count, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
            )
            wp.copy(baked, value)
            return baked
        return wp.full(
            self._robot_count,
            float(value),
            dtype=wp.float32,
            device=self._device,
            requires_grad=self._requires_grad,
        )

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
    def command_type(self) -> CommandType:
        return self._command_type

    @property
    def task_dim(self) -> int:
        return self._task_dim

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_graphable(self) -> bool:
        return self._ik_method not in _NON_CAPTURABLE_METHODS

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` with zero-initialised arrays."""
        dev, rg, n = self._device, self._requires_grad, self._robot_count
        inputs = ControllerDifferentialKinematicsModelFree.Inputs()
        inputs.joint_q = wp.zeros(self._min_len_q, dtype=wp.float32, device=dev, requires_grad=rg)
        inputs.ee_pos = wp.zeros(n, dtype=wp.vec3, device=dev, requires_grad=rg)
        inputs.ee_quat = wp.zeros(n, dtype=wp.quatf, device=dev, requires_grad=rg)
        inputs.target_pos = wp.zeros(n, dtype=wp.vec3, device=dev, requires_grad=rg)
        inputs.target_quat = wp.zeros(n, dtype=wp.quatf, device=dev, requires_grad=rg) if self._task_dim == 6 else None
        inputs.jacobian = wp.zeros((n, self._task_dim, self._max_dofs), dtype=wp.float32, device=dev, requires_grad=rg)
        inputs.bandwidth = (
            wp.zeros(n, dtype=wp.float32, device=dev, requires_grad=rg) if self._bandwidth_baked is None else None
        )
        inputs.solver_damping = (
            wp.zeros(n, dtype=wp.float32, device=dev, requires_grad=rg) if self._damping_baked is None else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with flat target arrays."""
        dev, rg = self._device, self._requires_grad
        outputs = ControllerDifferentialKinematicsModelFree.Outputs()
        outputs.joint_target_q = wp.zeros(self._min_len_target_q, dtype=wp.float32, device=dev, requires_grad=rg)
        outputs.joint_target_qd = wp.zeros(self._min_len_target_qd, dtype=wp.float32, device=dev, requires_grad=rg)
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Run one differential-IK step.

        Args:
            inputs: Populated :class:`Inputs` struct. The Jacobian must use
                Newton's COM-referenced convention.
            outputs: :class:`Outputs` struct to write targets into.
            dt: Control period [s], used to integrate the velocity command into
                a position target. Pass a 1-element array to vary it under graph
                replay; a float is recorded at capture time.
        """
        bandwidth = self._bandwidth_baked if self._bandwidth_baked is not None else inputs.bandwidth
        damping = self._damping_baked if self._damping_baked is not None else inputs.solver_damping
        n, task_dim = self._robot_count, self._task_dim

        self._validate_ports(inputs, outputs, bandwidth, damping)

        if isinstance(dt, wp.array):
            _validate_with_exact_shape(array=dt, name="dt", dtype=wp.float32, shape=(1,), device=self._device)
            dt_buf = dt
        else:
            self._dt_buf.fill_(float(dt))
            dt_buf = self._dt_buf

        wp.launch(
            _gather_flat_kernel,
            dim=self._controlled_dof_count,
            inputs=[inputs.joint_q, self._q_idx],
            outputs=[self._q_flat],
            device=self._device,
        )

        if task_dim == 3:
            wp.launch(
                _compute_position_error_kernel,
                dim=n,
                inputs=[inputs.ee_pos, inputs.target_pos],
                outputs=[self._e_buffer],
                device=self._device,
            )
        else:
            wp.launch(
                _compute_pose_error_kernel,
                dim=n,
                inputs=[inputs.ee_pos, inputs.ee_quat, inputs.target_pos, inputs.target_quat],
                outputs=[self._e_buffer],
                device=self._device,
            )

        j_site = inputs.jacobian
        if self._orientation_weight is not None:
            wx, wy, wz = self._orientation_weight
            wp.launch(
                _copy_jacobian_kernel,
                dim=(n, task_dim, self._max_dofs),
                inputs=[j_site],
                outputs=[self._j_internal],
                device=self._device,
            )
            j_site = self._j_internal
            wp.launch(
                _scale_orient_jacobian_rows_kernel,
                dim=(n, self._max_dofs),
                inputs=[j_site, wx, wy, wz],
                device=self._device,
            )
            wp.launch(_scale_orient_error_rows_kernel, dim=n, inputs=[self._e_buffer, wx, wy, wz], device=self._device)

        run_solver(
            self._ik_method,
            j_site,
            self._e_buffer,
            bandwidth,
            self._zero_damping if self._ik_method == IkMethod.PSEUDO_INVERSE else damping,
            self._A,
            self._y,
            self._qd_padded,
            n,
            self._max_dofs,
            task_dim,
            self._min_singular_value,
            self._lambda_min,
            self._lambda_max,
            self._sigma_thresh,
            self._power_iter_v,
            self._sigma_min_est,
            self._adaptive_damping,
            self._lambda_min_array,
            self._device,
        )

        if self._avoidance_gain > 0.0:
            wp.launch(
                _null_space_correction_kernel,
                dim=n,
                inputs=[
                    j_site,
                    self._q_flat,
                    self._joint_pos_lower,
                    self._joint_pos_upper,
                    self._dof_offsets,
                    self._avoidance_gain,
                    self._avoidance_margin,
                ],
                outputs=[self._qd_padded],
                device=self._device,
            )

        wp.launch(
            _compact_qd_kernel,
            dim=(n, self._max_dofs),
            inputs=[self._qd_padded, self._dof_offsets],
            outputs=[self._qd_flat],
            device=self._device,
        )
        wp.launch(
            _integrate_position_kernel,
            dim=self._controlled_dof_count,
            inputs=[self._qd_flat, self._q_flat, dt_buf],
            outputs=[self._q_out_flat],
            device=self._device,
        )
        wp.launch(
            _scatter_flat_kernel,
            dim=self._controlled_dof_count,
            inputs=[self._q_out_flat, self._target_q_idx],
            outputs=[outputs.joint_target_q],
            device=self._device,
        )
        wp.launch(
            _scatter_flat_kernel,
            dim=self._controlled_dof_count,
            inputs=[self._qd_flat, self._target_qd_idx],
            outputs=[outputs.joint_target_qd],
            device=self._device,
        )

    def _validate_ports(self, inputs: Inputs, outputs: Outputs, bandwidth, damping) -> None:
        """Check every port bound by the caller before any kernel reads it."""
        n = self._robot_count
        _validate_with_minimum_shape(
            array=inputs.joint_q, dtype=wp.float32, name="inputs.joint_q", shape=(self._min_len_q,), device=self._device
        )
        _validate_with_minimum_shape(
            array=outputs.joint_target_q,
            dtype=wp.float32,
            name="outputs.joint_target_q",
            shape=(self._min_len_target_q,),
            device=self._device,
        )
        _validate_with_minimum_shape(
            array=outputs.joint_target_qd,
            dtype=wp.float32,
            name="outputs.joint_target_qd",
            shape=(self._min_len_target_qd,),
            device=self._device,
        )
        for name, array, dtype in (
            ("inputs.ee_pos", inputs.ee_pos, wp.vec3),
            ("inputs.target_pos", inputs.target_pos, wp.vec3),
        ):
            _validate_with_exact_shape(array=array, name=name, dtype=dtype, shape=(n,), device=self._device)
        if self._task_dim == 6:
            for name, array in (("inputs.ee_quat", inputs.ee_quat), ("inputs.target_quat", inputs.target_quat)):
                _validate_with_exact_shape(array=array, name=name, dtype=wp.quatf, shape=(n,), device=self._device)
        _validate_with_exact_shape(
            array=inputs.jacobian,
            name="inputs.jacobian",
            dtype=wp.float32,
            shape=(n, self._task_dim, self._max_dofs),
            device=self._device,
        )
        _validate_with_exact_shape(array=bandwidth, name="bandwidth", dtype=wp.float32, shape=(n,), device=self._device)
        if self._ik_method != IkMethod.TRANSPOSE:
            _validate_with_exact_shape(
                array=damping, name="solver_damping", dtype=wp.float32, shape=(n,), device=self._device
            )


def _normalize_orientation_weight(
    orientation_weight: float | tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """Expand an orientation weight to a per-axis triple, or ``None`` if unused."""
    if orientation_weight is None:
        return None
    if isinstance(orientation_weight, (int, float)):
        w = float(orientation_weight)
        return (w, w, w)
    if len(orientation_weight) != 3:
        raise ValueError(f"orientation_weight must be a scalar or a length-3 tuple, got {orientation_weight!r}.")
    return tuple(float(v) for v in orientation_weight)
