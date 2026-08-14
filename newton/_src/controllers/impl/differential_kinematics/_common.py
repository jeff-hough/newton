# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared kernels and solver dispatch for the differential-IK controllers.

Joint-space vectors are flat, indexed by controlled DOF, with robot boundaries
carried by a CSR-style ``dof_offsets`` array. The Jacobian and the solver's
per-robot scratch stay blocked and padded to ``max_dofs``: they are operators,
not joint-space vectors.

Padding columns of the Jacobian are never written and so remain zero, which is
what lets the solver ignore ragged DOF counts entirely -- ``J J^T`` and ``J^T y``
both sum over a column that contributes nothing. Only the null-space correction,
which reads per-DOF joint limits, needs the real per-robot count.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import warp as wp

from ...kernels import _gather_flat_kernel, _scatter_flat_kernel

__all__ = [
    "DEFAULT_SOLVER_DAMPING",
    "_NON_CAPTURABLE_METHODS",
    "CommandType",
    "IkMethod",
    "_compact_qd_kernel",
    "_gather_flat_kernel",
    "_integrate_position_kernel",
    "_null_space_correction_kernel",
    "_quat_to_axis_angle",
    "_scale_orient_error_rows_kernel",
    "_scale_orient_jacobian_rows_kernel",
    "_scatter_flat_kernel",
    "run_solver",
]


class IkMethod(IntEnum):
    """Jacobian inverse strategy."""

    DAMPED_LEAST_SQUARES = 0
    TRANSPOSE = 1
    PSEUDO_INVERSE = 2
    SVD = 3
    ADAPTIVE_DAMPED_LEAST_SQUARES = 4


class CommandType(IntEnum):
    """Task-space command type for differential IK.

    Determines which error components are used and the Jacobian shape:

    - ``POSITION``: 3-DOF position control. Orientation error and rows are ignored.
    - ``POSE``: 6-DOF full pose control (position + orientation).

    The integer value equals the task-space dimension (3 or 6).
    """

    POSITION = 3
    POSE = 6


DEFAULT_SOLVER_DAMPING: float = 0.05

# SVD calls into ``torch.linalg`` and cannot be captured in a CUDA graph
_NON_CAPTURABLE_METHODS = frozenset([IkMethod.SVD])


@wp.func
def _quat_to_axis_angle(q_err: wp.quat) -> wp.vec3:
    """Return the axis-angle error ``θ·axis`` of an error quaternion.

    Matches Isaac Lab's ``axis_angle_from_quat``: ``q_err.xyz / (sin(θ/2) / θ)``,
    with a Taylor fallback for small angles and double-cover sign normalisation.
    """
    # double-cover: ensure w >= 0
    sign = float(1.0)
    if q_err[3] < float(0.0):
        sign = float(-1.0)
    qx = sign * q_err[0]
    qy = sign * q_err[1]
    qz = sign * q_err[2]
    qw = sign * q_err[3]

    mag = wp.sqrt(qx * qx + qy * qy + qz * qz)
    half_angle = wp.atan2(mag, qw)
    angle = float(2.0) * half_angle

    eps = float(1.0e-6)
    sin_ha_over_angle = float(0.0)
    if wp.abs(angle) > eps:
        sin_ha_over_angle = wp.sin(half_angle) / angle
    else:
        # Taylor: sin(θ/2)/θ ≈ 0.5 - θ²/48
        sin_ha_over_angle = float(0.5) - angle * angle / float(48.0)

    return wp.vec3(qx / sin_ha_over_angle, qy / sin_ha_over_angle, qz / sin_ha_over_angle)


# ---------------------------------------------------------------------------
# The one Jacobian-inverse solve: θ̇ = bandwidth · J^T (J J^T + λ²I)^-1 e
#
# Used regardless of DOF count versus task dimension: for any λ > 0 this is
# algebraically identical to the joint-space form (J^TJ + λ²I)^-1J^T, and J J^T is
# always task_dimxtask_dim (<= 6x6), so it covers every case without a runtime
# shape branch. The real block occupies the top-left task_dimxtask_dim corner;
# remaining diagonal entries are 1 so one 6x6 Cholesky kernel serves both task
# dimensions.
# ---------------------------------------------------------------------------


@wp.kernel
def _build_jjt_matrix_kernel(
    j_site: wp.array3d[wp.float32],  # (N, task_dim, max_dofs)
    solver_damping: wp.array[wp.float32],  # (N,)
    max_dofs: int,
    task_dim: int,
    A: wp.array3d[wp.float32],  # (N, 6, 6)
):
    """Build ``J J^T + λ_eff²I`` in the top-left block, identity padding elsewhere.

    ``λ_eff²`` floors the requested damping at a small fraction of
    ``trace(J J^T)``: PSEUDO_INVERSE passes ``λ=0``, and an un-floored Cholesky
    pivot can be non-positive at a rank-deficient Jacobian. The floor is
    negligible for any method that already damps above it, and a task direction
    ``J`` cannot reach at all is multiplied by an exactly-zero row of ``J^T``
    downstream, so it does not perturb the result.

    Summing over ``max_dofs`` is safe for ragged fleets because a short robot's
    padding columns are never written and stay zero.
    """
    r = wp.tid()
    lam = solver_damping[r]

    trace = float(0.0)
    for i in range(task_dim):
        diag = float(0.0)
        for j in range(max_dofs):
            diag = diag + j_site[r, i, j] * j_site[r, i, j]
        trace = trace + diag

    floor = float(1.0e-9) * trace / float(task_dim)
    lam_sq = wp.max(lam * lam, floor)

    for i in range(6):
        for k in range(6):
            if i < task_dim and k < task_dim:
                acc = float(0.0)
                for j in range(max_dofs):
                    acc = acc + j_site[r, i, j] * j_site[r, k, j]
                if i == k:
                    acc = acc + lam_sq
                A[r, i, k] = acc
            elif i == k:
                A[r, i, k] = float(1.0)
            else:
                A[r, i, k] = float(0.0)


# wp.tile_cholesky adjoints return zero gradients in current Warp; marking
# enable_backward=False makes that explicit rather than silently wrong.
@wp.kernel(enable_backward=False)
def _cholesky_solve_kernel(
    A: wp.array3d[wp.float32],
    e_buffer: wp.array2d[wp.float32],
    y: wp.array2d[wp.float32],
):
    """Solve the padded 6x6 system ``A y = e`` via Cholesky factorisation."""
    r = wp.tid()
    A_tile = wp.tile_load(A[r], shape=(6, 6))
    e_tile = wp.tile_load(e_buffer[r], shape=(6,))
    L = wp.tile_cholesky(A_tile)
    y_tile = wp.tile_cholesky_solve(L, e_tile)
    wp.tile_store(y[r], y_tile)


@wp.kernel
def _qd_from_y_kernel(
    j_site: wp.array3d[wp.float32],
    y: wp.array2d[wp.float32],
    bandwidth: wp.array[wp.float32],
    task_dim: int,
    qd_padded: wp.array2d[wp.float32],  # (N, max_dofs)
):
    """``θ̇ = bandwidth · J^T y``, summed over the task-space rows."""
    r, j = wp.tid()
    val = float(0.0)
    for i in range(task_dim):
        val = val + j_site[r, i, j] * y[r, i]
    qd_padded[r, j] = bandwidth[r] * val


@wp.kernel
def _qd_from_jte_kernel(
    j_site: wp.array3d[wp.float32],
    e_buffer: wp.array2d[wp.float32],
    bandwidth: wp.array[wp.float32],
    task_dim: int,
    qd_padded: wp.array2d[wp.float32],  # (N, max_dofs)
):
    """``θ̇ = bandwidth · J^T e``, summed over the task-space rows."""
    r, j = wp.tid()
    val = float(0.0)
    for i in range(task_dim):
        val = val + j_site[r, i, j] * e_buffer[r, i]
    qd_padded[r, j] = bandwidth[r] * val


# ---------------------------------------------------------------------------
# Orientation weighting (in-place, capturable)
# ---------------------------------------------------------------------------


@wp.kernel
def _scale_orient_jacobian_rows_kernel(
    j_site: wp.array3d[wp.float32],
    wx: float,
    wy: float,
    wz: float,
):
    """Scale the three orientation rows of ``j_site`` in place."""
    r, j = wp.tid()
    j_site[r, 3, j] = j_site[r, 3, j] * wx
    j_site[r, 4, j] = j_site[r, 4, j] * wy
    j_site[r, 5, j] = j_site[r, 5, j] * wz


@wp.kernel
def _scale_orient_error_rows_kernel(
    e_buffer: wp.array2d[wp.float32],
    wx: float,
    wy: float,
    wz: float,
):
    """Scale the three orientation entries of ``e_buffer`` in place."""
    r = wp.tid()
    e_buffer[r, 3] = e_buffer[r, 3] * wx
    e_buffer[r, 4] = e_buffer[r, 4] * wy
    e_buffer[r, 5] = e_buffer[r, 5] * wz


# ---------------------------------------------------------------------------
# ADAPTIVE_DAMPED_LEAST_SQUARES — sigma_min by inverse power iteration
# ---------------------------------------------------------------------------

_ADAPTIVE_POWER_ITERATIONS = 3


@wp.kernel
def _init_power_iteration_vector_kernel(
    task_dim: int,
    v: wp.array2d[wp.float32],  # (N, 6)
):
    """Seed the iterate uniformly over real task rows and zero over padding.

    The padded rows of ``A`` form an identity block decoupled from the real
    task block, so a seed that starts at zero there stays there — the padding's
    eigenvalue of 1 cannot contaminate the estimate.
    """
    r, i = wp.tid()
    if i < task_dim:
        v[r, i] = float(1.0) / wp.sqrt(float(task_dim))
    else:
        v[r, i] = float(0.0)


@wp.kernel
def _estimate_sigma_min_kernel(
    A: wp.array3d[wp.float32],  # (N, 6, 6) — J J^T + λ_min²I, padded
    v0: wp.array2d[wp.float32],  # (N, 6)
    lambda_min: float,
    sigma_min_est: wp.array[wp.float32],  # (N,)
):
    """Estimate ``sigma_min(J)`` by inverse power iteration on ``A``.

    At convergence ``‖A^-1v‖ → 1/λ_min(A)`` for the unit eigenvector of ``A``'s
    smallest eigenvalue, so ``sigma_min(J) = sqrt(λ_min(A) - λ_min²)``. Three
    iterations suffice at or below the default threshold; above it the ratio
    derived from the estimate saturates and the error is discarded.
    """
    r = wp.tid()
    A_tile = wp.tile_load(A[r], shape=(6, 6))
    v_tile = wp.tile_load(v0[r], shape=(6,))
    L = wp.tile_cholesky(A_tile)

    norm = float(1.0)
    for _ in range(_ADAPTIVE_POWER_ITERATIONS):
        solved = wp.tile_cholesky_solve(L, v_tile)
        norm_sq_tile = wp.tile_dot(solved, solved)
        norm = wp.sqrt(norm_sq_tile[0])
        v_tile = solved / norm

    lam_min_A = float(1.0) / norm
    sigma_sq = wp.max(lam_min_A - lambda_min * lambda_min, float(0.0))
    sigma_min_est[r] = wp.sqrt(sigma_sq)


@wp.kernel
def _adaptive_damping_kernel(
    sigma_min_est: wp.array[wp.float32],
    lambda_min: float,
    lambda_max: float,
    sigma_thresh: float,
    damping_out: wp.array[wp.float32],  # (N,) λ, not λ²
):
    """Maciejewski-Klein damping schedule from the estimated ``sigma_min``."""
    r = wp.tid()
    ratio = wp.clamp(sigma_min_est[r] / sigma_thresh, float(0.0), float(1.0))
    lam_sq = lambda_min * lambda_min + (float(1.0) - ratio * ratio) * (
        lambda_max * lambda_max - lambda_min * lambda_min
    )
    damping_out[r] = wp.sqrt(lam_sq)


# ---------------------------------------------------------------------------
# Ragged-fleet plumbing
# ---------------------------------------------------------------------------


@wp.kernel
def _compact_qd_kernel(
    qd_padded: wp.array2d[wp.float32],  # (N, max_dofs)
    dof_offsets: wp.array[wp.int32],  # (N + 1,)
    qd_flat: wp.array[wp.float32],  # (controlled_dof_count,)
):
    """Compact the solver's padded per-robot result into the flat DOF layout."""
    robot, j = wp.tid()
    begin = dof_offsets[robot]
    if j >= dof_offsets[robot + 1] - begin:
        return
    qd_flat[begin + j] = qd_padded[robot, j]


@wp.kernel
def _integrate_position_kernel(
    qd_target: wp.array[wp.float32],  # (controlled_dof_count,)
    joint_q: wp.array[wp.float32],
    dt_buf: wp.array[wp.float32],  # (1,)
    joint_target_q: wp.array[wp.float32],
):
    """Integrate the velocity command forward one control period."""
    dof = wp.tid()
    joint_target_q[dof] = joint_q[dof] + qd_target[dof] * dt_buf[0]


# ---------------------------------------------------------------------------
# Null-space joint-limit avoidance — pure Warp, capturable
# ---------------------------------------------------------------------------


@wp.kernel
def _null_space_correction_kernel(
    j_site: wp.array3d[wp.float32],  # (N, task_dim, max_dofs) — rows 0:3 used
    joint_q: wp.array[wp.float32],  # (controlled_dof_count,)
    lower: wp.array[wp.float32],  # (controlled_dof_count,)
    upper: wp.array[wp.float32],  # (controlled_dof_count,)
    dof_offsets: wp.array[wp.int32],  # (N + 1,)
    gain: float,
    margin: float,
    qd_padded: wp.array2d[wp.float32],  # (N, max_dofs) — modified in place
):
    """Project a joint-centering bias into the null space of the position rows.

    Computes ``qd_null = (I - J_pos⁺ J_pos) dq_center`` and adds it, so the
    centering motion never perturbs the commanded end-effector position.
    ``J_pos⁺ = J_pos^T (J_pos J_pos^T)^-1`` comes from an in-register Cholesky
    factorisation of the 3x3 Gram matrix.

    Unlike the solver kernels this cannot ignore padding: the joint limits are
    per-DOF, so a short robot must not read past its own range.
    """
    r = wp.tid()
    begin = dof_offsets[r]
    dofs = dof_offsets[r + 1] - begin

    # -- Pass 1: build B = J_pos J_pos^T and v = J_pos dq_center.
    B00 = float(0.0)
    B01 = float(0.0)
    B02 = float(0.0)
    B11 = float(0.0)
    B12 = float(0.0)
    B22 = float(0.0)
    v0 = float(0.0)
    v1 = float(0.0)
    v2 = float(0.0)

    for j in range(dofs):
        p0 = j_site[r, 0, j]
        p1 = j_site[r, 1, j]
        p2 = j_site[r, 2, j]
        q = joint_q[begin + j]
        lo = lower[begin + j]
        hi = upper[begin + j]
        q_mid = float(0.5) * (lo + hi)
        dist = wp.min(q - lo, hi - q)
        activation = float(1.0) - wp.clamp(dist / margin, float(0.0), float(1.0))
        dq_j = -gain * activation * (q - q_mid)
        B00 = B00 + p0 * p0
        B01 = B01 + p0 * p1
        B02 = B02 + p0 * p2
        B11 = B11 + p1 * p1
        B12 = B12 + p1 * p2
        B22 = B22 + p2 * p2
        v0 = v0 + p0 * dq_j
        v1 = v1 + p1 * dq_j
        v2 = v2 + p2 * dq_j

    # -- Cholesky B = L L^T, with an eps floor for near-singular configurations.
    eps = float(1.0e-10)
    L00 = wp.sqrt(wp.max(B00, eps))
    L10 = B01 / L00
    L20 = B02 / L00
    L11 = wp.sqrt(wp.max(B11 - L10 * L10, eps))
    L21 = (B12 - L20 * L10) / L11
    L22 = wp.sqrt(wp.max(B22 - L20 * L20 - L21 * L21, eps))

    # -- Solve L y = v then L^T w = y, giving w = B^-1 v.
    y0 = v0 / L00
    y1 = (v1 - L10 * y0) / L11
    y2 = (v2 - L20 * y0 - L21 * y1) / L22
    w2 = y2 / L22
    w1 = (y1 - L21 * w2) / L11
    w0 = (y0 - L10 * w1 - L20 * w2) / L00

    # -- Pass 2: add dq_center minus its projection onto the row space.
    for j in range(dofs):
        p0 = j_site[r, 0, j]
        p1 = j_site[r, 1, j]
        p2 = j_site[r, 2, j]
        q = joint_q[begin + j]
        lo = lower[begin + j]
        hi = upper[begin + j]
        q_mid = float(0.5) * (lo + hi)
        dist = wp.min(q - lo, hi - q)
        activation = float(1.0) - wp.clamp(dist / margin, float(0.0), float(1.0))
        dq_j = -gain * activation * (q - q_mid)
        qd_padded[r, j] = qd_padded[r, j] + dq_j - (p0 * w0 + p1 * w1 + p2 * w2)


# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------


def run_solver(
    ik_method: IkMethod,
    j_site: wp.array3d[wp.float32],
    e_buffer: wp.array2d[wp.float32],  # (N, 6); task_dim entries real, rest zero
    bandwidth: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    A: wp.array3d[wp.float32],  # (N, 6, 6); identity-padded outside the task block
    y: wp.array2d[wp.float32],  # (N, 6)
    qd_padded: wp.array2d[wp.float32],  # (N, max_dofs)
    num_robots: int,
    max_dofs: int,
    task_dim: int,
    min_singular_value: float,
    lambda_min: float,
    lambda_max: float,
    sigma_thresh: float,
    power_iter_v: wp.array2d[wp.float32] | None,
    sigma_min_est: wp.array[wp.float32] | None,
    adaptive_damping: wp.array[wp.float32] | None,
    lambda_min_array: wp.array[wp.float32] | None,
    device: Any,
) -> None:
    """Fill ``qd_padded`` using the chosen inverse-Jacobian method.

    DAMPED_LEAST_SQUARES and PSEUDO_INVERSE resolve to the same minimum-norm
    solve, padded to a fixed 6x6 Cholesky system. TRANSPOSE is a simpler kernel
    with no factorisation. ADAPTIVE_DAMPED_LEAST_SQUARES estimates ``sigma_min``
    by inverse power iteration and reuses the same solve. All of these are
    graph-capturable; SVD calls ``torch.linalg`` and is not.
    """
    if ik_method in (IkMethod.DAMPED_LEAST_SQUARES, IkMethod.PSEUDO_INVERSE):
        # PSEUDO_INVERSE passes a zero damping array; the kernel floors λ².
        wp.launch(
            _build_jjt_matrix_kernel,
            dim=num_robots,
            inputs=[j_site, damping, max_dofs, task_dim],
            outputs=[A],
            device=device,
        )
        wp.launch_tiled(
            _cholesky_solve_kernel, dim=[num_robots], inputs=[A, e_buffer], outputs=[y], block_dim=32, device=device
        )
        wp.launch(
            _qd_from_y_kernel,
            dim=(num_robots, max_dofs),
            inputs=[j_site, y, bandwidth, task_dim],
            outputs=[qd_padded],
            device=device,
        )

    elif ik_method == IkMethod.TRANSPOSE:
        wp.launch(
            _qd_from_jte_kernel,
            dim=(num_robots, max_dofs),
            inputs=[j_site, e_buffer, bandwidth, task_dim],
            outputs=[qd_padded],
            device=device,
        )

    elif ik_method == IkMethod.SVD:
        import torch

        j_t = wp.to_torch(j_site)  # (N, task_dim, max_dofs)
        e_t = wp.to_torch(e_buffer)[:, :task_dim]  # (N, task_dim)
        bw_t = wp.to_torch(bandwidth)
        U, S, Vh = torch.linalg.svd(j_t, full_matrices=False)
        S_inv = torch.where(S > min_singular_value, 1.0 / S, torch.zeros_like(S))
        j_pinv = Vh.mT @ torch.diag_embed(S_inv) @ U.mT  # (N, max_dofs, task_dim)
        delta_q = torch.bmm(j_pinv, e_t.unsqueeze(-1)).squeeze(-1)
        wp.to_torch(qd_padded)[:] = bw_t.unsqueeze(-1) * delta_q

    elif ik_method == IkMethod.ADAPTIVE_DAMPED_LEAST_SQUARES:
        wp.launch(
            _build_jjt_matrix_kernel,
            dim=num_robots,
            inputs=[j_site, lambda_min_array, max_dofs, task_dim],
            outputs=[A],
            device=device,
        )
        wp.launch(
            _init_power_iteration_vector_kernel,
            dim=(num_robots, 6),
            inputs=[task_dim],
            outputs=[power_iter_v],
            device=device,
        )
        wp.launch_tiled(
            _estimate_sigma_min_kernel,
            dim=[num_robots],
            inputs=[A, power_iter_v, lambda_min],
            outputs=[sigma_min_est],
            block_dim=32,
            device=device,
        )
        wp.launch(
            _adaptive_damping_kernel,
            dim=num_robots,
            inputs=[sigma_min_est, lambda_min, lambda_max, sigma_thresh],
            outputs=[adaptive_damping],
            device=device,
        )
        wp.launch(
            _build_jjt_matrix_kernel,
            dim=num_robots,
            inputs=[j_site, adaptive_damping, max_dofs, task_dim],
            outputs=[A],
            device=device,
        )
        wp.launch_tiled(
            _cholesky_solve_kernel, dim=[num_robots], inputs=[A, e_buffer], outputs=[y], block_dim=32, device=device
        )
        wp.launch(
            _qd_from_y_kernel,
            dim=(num_robots, max_dofs),
            inputs=[j_site, y, bandwidth, task_dim],
            outputs=[qd_padded],
            device=device,
        )

    else:
        raise ValueError(f"Unknown IkMethod: {ik_method}")
