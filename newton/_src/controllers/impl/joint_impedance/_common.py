# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for :class:`~newton.controllers.ControllerJointImpedance`.

Joint-space vectors are flat, indexed by controlled DOF. Robot boundaries are
carried by a CSR-style ``dof_offsets`` array of length ``robot_count + 1``, so
only the mass matrix -- an operator, not a joint-space vector -- is blocked and
padded per robot.

The generic gather/scatter kernels live in :mod:`newton._src.controllers.kernels`
and are re-exported here so both controllers share one implementation.
"""

import warp as wp

from ...kernels import _gather_flat_kernel, _scatter_flat_kernel

__all__ = [
    "_add_term_kernel",
    "_gather_flat_kernel",
    "_gather_mass_matrix_blocks_kernel",
    "_mass_matrix_multiply_kernel",
    "_pd_term_kernel",
    "_scatter_flat_kernel",
]


@wp.kernel
def _pd_term_kernel(
    joint_q: wp.array[wp.float32],  # (controlled_dof_count,)
    joint_qd: wp.array[wp.float32],
    joint_q_des: wp.array[wp.float32],
    joint_qd_des: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    dof = wp.tid()
    out[dof] = stiffness[dof] * (joint_q_des[dof] - joint_q[dof]) + damping[dof] * (joint_qd_des[dof] - joint_qd[dof])


@wp.kernel
def _add_term_kernel(
    term: wp.array[wp.float32],  # (controlled_dof_count,)
    tau: wp.array[wp.float32],
):
    dof = wp.tid()
    tau[dof] = tau[dof] + term[dof]


@wp.kernel
def _mass_matrix_multiply_kernel(
    M: wp.array3d[wp.float32],  # (robot_count, max_dofs, max_dofs)
    vec: wp.array[wp.float32],  # (controlled_dof_count,)
    dof_offsets: wp.array[wp.int32],  # (robot_count + 1,)
    out: wp.array[wp.float32],  # (controlled_dof_count,)
):
    robot, row = wp.tid()
    begin = dof_offsets[robot]
    dofs = dof_offsets[robot + 1] - begin
    if row >= dofs:
        return
    acc = float(0.0)
    for col in range(dofs):
        acc = acc + M[robot, row, col] * vec[begin + col]
    out[begin + row] = acc


@wp.kernel
def _gather_mass_matrix_blocks_kernel(
    mass_matrix_full: wp.array3d[wp.float32],  # (articulation_count, model_max_dofs, model_max_dofs)
    selected_articulations: wp.array[wp.int32],  # (robot_count,) robot -> model articulation
    controlled_dof_to_model_dof: wp.array[wp.int32],  # (controlled_dof_count,)
    dof_offsets: wp.array[wp.int32],  # (robot_count + 1,)
    articulation_dof_start: wp.array[wp.int32],  # (robot_count,) first model DOF of each robot
    mass_matrix_selected: wp.array3d[wp.float32],  # (robot_count, max_dofs, max_dofs)
):
    robot, row, col = wp.tid()
    begin = dof_offsets[robot]
    dofs = dof_offsets[robot + 1] - begin
    if row >= dofs or col >= dofs:
        return
    # H is indexed by DOF within the articulation, so shift out of model DOF space.
    base = articulation_dof_start[robot]
    mass_matrix_selected[robot, row, col] = mass_matrix_full[
        selected_articulations[robot],
        controlled_dof_to_model_dof[begin + row] - base,
        controlled_dof_to_model_dof[begin + col] - base,
    ]
