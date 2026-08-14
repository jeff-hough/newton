# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialKinematics — one-step differential IK from a Newton model.

Runs :func:`newton.eval_fk` and :func:`newton.eval_jacobian` on the supplied
model each step, shifts the COM-referenced Jacobian to the controlled site, and
delegates the error, solver, orientation weighting, and null-space avoidance to
an inner :class:`ControllerDifferentialKinematicsModelFree` instance, so both
classes share identical math.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from newton._src.sim.articulation import eval_fk, eval_jacobian
from newton._src.sim.model import Model

from ...controller import ControllerBase
from ...dof_indices import _build_dof_indices
from ...kernels import _gather_flat_kernel
from ...utils import _validate_articulations_are_world_rooted, _validate_with_exact_shape, _validate_with_minimum_shape
from ._common import DEFAULT_SOLVER_DAMPING, CommandType, IkMethod
from .model_free import ControllerDifferentialKinematicsModelFree


@wp.kernel
def _site_pose_kernel(
    body_q: wp.array[wp.transform],
    site_body: wp.array[wp.int32],  # (N,) model body carrying the site
    site_xform: wp.array[wp.transform],  # (N,) site pose in that body's frame
    ee_pos: wp.array[wp.vec3],  # (N,) output
    ee_quat: wp.array[wp.quatf],  # (N,) output
):
    """Write the world-frame pose of each robot's controlled site."""
    r = wp.tid()
    t_site = body_q[site_body[r]] * site_xform[r]
    ee_pos[r] = wp.transform_get_translation(t_site)
    ee_quat[r] = wp.transform_get_rotation(t_site)


@wp.kernel
def _site_jacobian_kernel(
    jacobian: wp.array3d[wp.float32],  # (articulation_count, max_links*6, model_max_dofs)
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    selected_articulations: wp.array[wp.int32],  # (N,)
    site_body: wp.array[wp.int32],  # (N,) model body index
    site_link_row: wp.array[wp.int32],  # (N,) 6 * link index within the articulation
    site_xform: wp.array[wp.transform],  # (N,)
    controlled_dof_to_model_dof: wp.array[wp.int32],  # (controlled_dof_count,)
    dof_offsets: wp.array[wp.int32],  # (N + 1,)
    articulation_dof_start: wp.array[wp.int32],  # (N,)
    j_site: wp.array3d[wp.float32],  # (N, 6, max_dofs) output
):
    """Select the controlled columns and shift the Jacobian from the COM to the site.

    ``eval_jacobian`` produces columns indexed by DOF within the articulation,
    so a controlled DOF's column is found by shifting out of model DOF space —
    the same arithmetic the mass-matrix block gather uses. Padding columns of a
    short robot are never written and stay zero, which is what lets the solver
    ignore ragged DOF counts.
    """
    r, k = wp.tid()
    begin = dof_offsets[r]
    if k >= dof_offsets[r + 1] - begin:
        return

    body = site_body[r]
    t_body = body_q[body]
    offset = wp.transform_get_translation(t_body * site_xform[r]) - wp.transform_point(t_body, body_com[body])

    art = selected_articulations[r]
    row = site_link_row[r]
    col = controlled_dof_to_model_dof[begin + k] - articulation_dof_start[r]

    jl_x = jacobian[art, row + 0, col]
    jl_y = jacobian[art, row + 1, col]
    jl_z = jacobian[art, row + 2, col]
    ja_x = jacobian[art, row + 3, col]
    ja_y = jacobian[art, row + 4, col]
    ja_z = jacobian[art, row + 5, col]

    # v_site = v_com + omega x offset
    j_site[r, 0, k] = jl_x + ja_y * offset[2] - ja_z * offset[1]
    j_site[r, 1, k] = jl_y + ja_z * offset[0] - ja_x * offset[2]
    j_site[r, 2, k] = jl_z + ja_x * offset[1] - ja_y * offset[0]
    j_site[r, 3, k] = ja_x
    j_site[r, 4, k] = ja_y
    j_site[r, 5, k] = ja_z


@wp.kernel
def _copy_position_rows_kernel(
    j_site: wp.array3d[wp.float32],  # (N, 6, max_dofs)
    j_site_pos: wp.array3d[wp.float32],  # (N, 3, max_dofs)
):
    """Copy the three position rows for position-only control."""
    r, k = wp.tid()
    j_site_pos[r, 0, k] = j_site[r, 0, k]
    j_site_pos[r, 1, k] = j_site[r, 1, k]
    j_site_pos[r, 2, k] = j_site[r, 2, k]


class ControllerDifferentialKinematics(ControllerBase):
    """One-step differential IK driving a site pose toward a target.

    For each selected articulation, solves for a joint-velocity command that
    moves a labelled site toward a target pose, then integrates it into a joint
    position target. Coupled per robot: the solution depends on that robot's
    whole configuration. Stateless.

    ``model`` is borrowed, not owned — it is never written to, and changes to it
    are visible to the controller immediately.

    ``model`` may contain far more than the controlled robots. Every model
    coordinate and DOF is read each step, so uncontrolled joints still
    contribute to forward kinematics and to the Jacobian. Targets are produced
    only for the selected joints.

    Only the **controlled** joints are restricted to 1-DOF revolute or
    prismatic. Uncontrolled joints may be of any type, including free and ball
    joints, so a floating base is read but never driven.

    Supports heterogeneous robot fleets — selected articulations may have
    different DOF counts. Joint-space vectors are flat and ragged; only the
    Jacobian is padded.

    .. important::
        An uncontrolled joint must still belong to an articulation to be read.
        :func:`newton.eval_fk` is evaluated per articulation, so a joint outside
        every articulation never updates its body transform and anything mounted
        on it sees a stale base.

    Args:
        model: :class:`~newton.Model` containing the robots to control.
        site: Shape label identifying the controlled site, as given to
            :meth:`~newton.ModelBuilder.add_site`. Each selected articulation
            must carry exactly one shape with this label. Pass several labels to
            accept any one of them, which lets a single controller span robot
            kinds whose sites are named differently -- importers name sites after
            the asset, so the label is not always yours to choose. A label
            matching only shapes outside the selection is allowed, so one set of
            names can be reused whatever subset of the fleet is controlled.
        articulations: Which articulations to control, as model indices or
            labels. ``None`` selects all.
        joints: Which joints to control, resolved within the selected
            articulations. ``None`` selects every scalar joint. Use
            :func:`~newton.controllers.select_joints` to exclude a joint from an
            otherwise fully controlled robot.
        model_coord_to_sim_coord: Maps each model coordinate to its slot in
            ``inputs.joint_q``, shape ``(model.joint_coord_count,)``. ``None``
            uses the identity.
        model_dof_to_sim_dof: Maps each model DOF to its slot in
            ``inputs.joint_qd`` and the outputs, shape
            ``(model.joint_dof_count,)``. Kept separate because Newton indexes
            positions and velocities differently.
        bandwidth: Per-robot scale on the solved joint velocity. Float, array of
            shape ``(robot_count,)``, or ``None`` to read it live each step.
        solver_damping: Per-robot damped least-squares λ, same format.
        ik_method: Jacobian inverse strategy, see :class:`IkMethod`.
        command_type: Full pose or position only, see :class:`CommandType`.
        min_singular_value: Singular-value floor, SVD only.
        lambda_min: Baseline damping for adaptive damped least squares.
        lambda_max: Maximum damping for adaptive damped least squares.
        sigma_thresh: Singular value at which adaptive damping starts ramping.
        orientation_weight: Scalar or per-axis weight on the orientation rows.
            Requires ``CommandType.POSE``.
        joint_limit_avoidance_gain: Gain on the null-space centring bias. ``0``
            disables it.
        joint_limit_avoidance_margin: Distance from a limit at which centring
            starts to act.
        joint_pos_lower: Lower joint limits, shape ``(controlled_dof_count,)``.
            Defaults to the model's own limits at the controlled DOFs.
        joint_pos_upper: Upper joint limits, same shape and default.
        requires_grad: Whether internal buffers need gradient support.
    """

    IkMethod = IkMethod
    CommandType = CommandType
    DEFAULT_SOLVER_DAMPING: float = DEFAULT_SOLVER_DAMPING

    class Inputs:
        """Input struct returned by :meth:`~ControllerDifferentialKinematics.input`.

        The Jacobian and end-effector pose are computed internally and do not
        appear here.
        """

        joint_q: wp.array[wp.float32]
        """Current joint positions [m or rad], in simulation coordinate order."""
        joint_qd: wp.array[wp.float32]
        """Current joint velocities [m/s or rad/s], in simulation DOF order."""
        target_pos: wp.array[wp.vec3]
        """Desired site position [m] in world frame, shape ``(robot_count,)``."""
        target_quat: wp.array[wp.quatf] | None
        """Desired site orientation ``[x, y, z, w]`` in world frame, shape ``(robot_count,)``. ``None`` unless ``command_type`` is POSE."""
        bandwidth: wp.array[wp.float32] | None
        """Per-robot velocity scale [dimensionless], shape ``(robot_count,)``. ``None`` when baked at construction."""
        solver_damping: wp.array[wp.float32] | None
        """Per-robot damped least-squares λ [dimensionless], shape ``(robot_count,)``. ``None`` when baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerDifferentialKinematics.output`."""

        joint_target_q: wp.array[wp.float32]
        """Integrated joint position target [m or rad], in the simulation's target layout."""
        joint_target_qd: wp.array[wp.float32]
        """Joint velocity command [m/s or rad/s], in simulation DOF order."""

    def __init__(
        self,
        model: Model,
        *,
        site: str | list[str],
        articulations: list[int] | list[str] | None = None,
        joints: list[int] | list[str] | None = None,
        model_coord_to_sim_coord: wp.array[wp.int32] | None = None,
        model_dof_to_sim_dof: wp.array[wp.int32] | None = None,
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
        requires_grad: bool = False,
    ):
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")

        # Resolves the selection and rejects non-scalar controlled joints.
        self._dof_indices = _build_dof_indices(model, articulations=articulations, joints=joints)
        idx = self._dof_indices
        selected = idx.selected_articulations.numpy()
        _validate_articulations_are_world_rooted(model, selected)

        self._device = model.device
        self._requires_grad = requires_grad
        self._model = model
        self._model_state = model.state()
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)
        self._task_dim = int(command_type)

        if model_coord_to_sim_coord is None:
            model_coord_to_sim_coord = wp.array(
                np.arange(self._coord_count, dtype=np.int32), dtype=wp.int32, device=self._device
            )
        if model_dof_to_sim_dof is None:
            model_dof_to_sim_dof = wp.array(
                np.arange(self._dof_count, dtype=np.int32), dtype=wp.int32, device=self._device
            )

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere else.
        # ------------------------------------------------------------------
        flat_shape = (idx.controlled_dof_count,)
        for name, array, shape, required in (
            ("model_coord_to_sim_coord", model_coord_to_sim_coord, (self._coord_count,), True),
            ("model_dof_to_sim_dof", model_dof_to_sim_dof, (self._dof_count,), True),
        ):
            _validate_with_exact_shape(
                array=array, name=name, dtype=wp.int32, shape=shape, device=self._device, required=required
            )
        for name, array in (("joint_pos_lower", joint_pos_lower), ("joint_pos_upper", joint_pos_upper)):
            _validate_with_exact_shape(
                array=array, name=name, dtype=wp.float32, shape=flat_shape, device=self._device, required=False
            )
        # ------------------------------------------------------------------

        self._model_coord_to_sim_coord = model_coord_to_sim_coord
        self._model_dof_to_sim_dof = model_dof_to_sim_dof
        self._min_len_q = int(np.max(model_coord_to_sim_coord.numpy())) + 1
        self._min_len_qd = int(np.max(model_dof_to_sim_dof.numpy())) + 1

        c2m_dof = idx.controlled_dof_to_model_dof.numpy()
        c2m_coord = idx.controlled_dof_to_model_coord.numpy()
        controlled_to_sim_dof = model_dof_to_sim_dof.numpy()[c2m_dof]
        controlled_to_sim_coord = model_coord_to_sim_coord.numpy()[c2m_coord]

        # joint_target_q follows the simulation's target layout, which is
        # coordinate-shaped or DOF-shaped depending on this model's snapshot of
        # newton.use_coord_layout_targets. joint_target_qd is always DOF-shaped.
        target_q_layout = controlled_to_sim_coord if model.use_coord_layout_targets else controlled_to_sim_dof

        # Port sizes and liveness are derived here rather than read back off the
        # inner controller: this class builds the index arrays, so it already
        # knows both, and asking the inner would mean allocating a whole struct
        # just to keep two fields of it.
        self._min_len_target_q = int(target_q_layout.max()) + 1
        self._min_len_target_qd = int(controlled_to_sim_dof.max()) + 1
        self._bandwidth_is_live = bandwidth is None
        self._damping_is_live = solver_damping is None

        # Joint limits default to the model's own, gathered at the controlled DOFs.
        if joint_limit_avoidance_gain > 0.0 and (joint_pos_lower is None or joint_pos_upper is None):
            if model.joint_limit_lower is None or model.joint_limit_upper is None:
                raise ValueError(
                    "joint_limit_avoidance_gain > 0 needs joint limits, and this model has none; "
                    "pass joint_pos_lower and joint_pos_upper explicitly."
                )
            if joint_pos_lower is None:
                joint_pos_lower = wp.array(
                    model.joint_limit_lower.numpy()[c2m_dof], dtype=wp.float32, device=self._device
                )
            if joint_pos_upper is None:
                joint_pos_upper = wp.array(
                    model.joint_limit_upper.numpy()[c2m_dof], dtype=wp.float32, device=self._device
                )

        site_body, site_link_row, site_xform = _resolve_site(model, site, selected)
        self._site_body = wp.array(site_body, dtype=wp.int32, device=self._device)
        self._site_link_row = wp.array(np.asarray(site_link_row) * 6, dtype=wp.int32, device=self._device)
        self._site_xform = wp.array(site_xform, dtype=wp.transform, device=self._device)

        # Performance only: skipping unselected articulations changes nothing,
        # since their results are never read.
        self._eval_mask: wp.array[wp.bool] | None = None
        if idx.robot_count < model.articulation_count:
            mask_np = np.zeros(model.articulation_count, dtype=bool)
            mask_np[selected] = True
            self._eval_mask = wp.array(mask_np, dtype=wp.bool, device=self._device)

        rg = requires_grad
        self._jacobian_full = wp.zeros(
            (model.articulation_count, model.max_joints_per_articulation * 6, model.max_dofs_per_articulation),
            dtype=wp.float32,
            device=self._device,
            requires_grad=rg,
        )
        self._joint_S_s = wp.zeros(self._dof_count, dtype=wp.spatial_vector, device=self._device, requires_grad=rg)
        self._j_site = wp.zeros(
            (idx.robot_count, 6, idx.max_dofs), dtype=wp.float32, device=self._device, requires_grad=rg
        )
        # Position-only control hands the solver a slimmer Jacobian; the site
        # kernel still needs all six rows to perform the COM-to-site shift.
        self._j_site_pos: wp.array3d[wp.float32] | None = None
        if self._task_dim == 3:
            self._j_site_pos = wp.zeros(
                (idx.robot_count, 3, idx.max_dofs), dtype=wp.float32, device=self._device, requires_grad=rg
            )
        self._ee_pos = wp.zeros(idx.robot_count, dtype=wp.vec3, device=self._device, requires_grad=rg)
        self._ee_quat = wp.zeros(idx.robot_count, dtype=wp.quatf, device=self._device, requires_grad=rg)

        dofs_per_robot = wp.array(np.diff(idx.dof_offsets.numpy()), dtype=wp.int32, device=self._device)
        self._model_free = ControllerDifferentialKinematicsModelFree(
            dofs_per_robot=dofs_per_robot,
            bandwidth=bandwidth,
            solver_damping=solver_damping,
            ik_method=ik_method,
            command_type=command_type,
            min_singular_value=min_singular_value,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            sigma_thresh=sigma_thresh,
            orientation_weight=orientation_weight,
            joint_limit_avoidance_gain=joint_limit_avoidance_gain,
            joint_limit_avoidance_margin=joint_limit_avoidance_margin,
            joint_pos_lower=joint_pos_lower,
            joint_pos_upper=joint_pos_upper,
            # The inner controller reads this controller's model state, so its
            # input port is indexed in model coordinate space; its outputs are
            # composed with the caller's mapping and land in simulation space.
            joint_q_idx=idx.controlled_dof_to_model_coord,
            joint_target_q_idx=wp.array(target_q_layout, dtype=wp.int32, device=self._device),
            joint_target_qd_idx=wp.array(controlled_to_sim_dof, dtype=wp.int32, device=self._device),
            device=self._device,
            requires_grad=requires_grad,
        )

        # Pre-wired fields forwarded to the inner controller each step.
        self._mf_input = ControllerDifferentialKinematicsModelFree.Inputs()
        self._mf_input.joint_q = self._model_state.joint_q
        self._mf_input.ee_pos = self._ee_pos
        self._mf_input.ee_quat = self._ee_quat
        self._mf_input.jacobian = self._j_site_pos if self._task_dim == 3 else self._j_site

    @property
    def robot_count(self) -> int:
        return self._dof_indices.robot_count

    @property
    def max_dofs(self) -> int:
        return self._dof_indices.max_dofs

    @property
    def controlled_dof_count(self) -> int:
        return self._dof_indices.controlled_dof_count

    @property
    def controlled_joints(self) -> tuple[int, ...]:
        """Model joint index of each controlled DOF, in controlled order."""
        return self._dof_indices.controlled_joints

    @property
    def command_type(self) -> CommandType:
        return self._model_free.command_type

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
        return self._model_free.is_graphable()

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` without Jacobian or pose fields."""
        dev, rg, n = self._device, self._requires_grad, self.robot_count
        inputs = ControllerDifferentialKinematics.Inputs()
        inputs.joint_q = wp.zeros(self._min_len_q, dtype=wp.float32, device=dev, requires_grad=rg)
        inputs.joint_qd = wp.zeros(self._min_len_qd, dtype=wp.float32, device=dev, requires_grad=rg)
        inputs.target_pos = wp.zeros(n, dtype=wp.vec3, device=dev, requires_grad=rg)
        inputs.target_quat = wp.zeros(n, dtype=wp.quatf, device=dev, requires_grad=rg) if self._task_dim == 6 else None
        inputs.bandwidth = (
            wp.zeros(n, dtype=wp.float32, device=dev, requires_grad=rg) if self._bandwidth_is_live else None
        )
        inputs.solver_damping = (
            wp.zeros(n, dtype=wp.float32, device=dev, requires_grad=rg) if self._damping_is_live else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with flat target arrays."""
        dev, rg = self._device, self._requires_grad
        outputs = ControllerDifferentialKinematics.Outputs()
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
            inputs: Populated :class:`Inputs` struct. The Jacobian and the site
                pose are computed internally from the Newton model.
            outputs: :class:`Outputs` struct to write targets into.
            dt: Control period [s], used to integrate the velocity command into
                a position target.
        """
        # Checked here because the gathers below read these two ports before
        # the inner controller — which validates the rest — ever sees them.
        _validate_with_minimum_shape(
            array=inputs.joint_q, dtype=wp.float32, name="inputs.joint_q", shape=(self._min_len_q,), device=self._device
        )
        _validate_with_minimum_shape(
            array=inputs.joint_qd,
            dtype=wp.float32,
            name="inputs.joint_qd",
            shape=(self._min_len_qd,),
            device=self._device,
        )

        idx = self._dof_indices
        n, max_dofs = idx.robot_count, idx.max_dofs

        # Populate every model coordinate and DOF, not just the controlled ones:
        # an uncontrolled joint upstream of a controlled articulation still sets
        # its base transform, and hence the site pose and the Jacobian.
        wp.launch(
            _gather_flat_kernel,
            dim=self._coord_count,
            inputs=[inputs.joint_q, self._model_coord_to_sim_coord],
            outputs=[self._model_state.joint_q],
            device=self._device,
        )
        wp.launch(
            _gather_flat_kernel,
            dim=self._dof_count,
            inputs=[inputs.joint_qd, self._model_dof_to_sim_dof],
            outputs=[self._model_state.joint_qd],
            device=self._device,
        )

        eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        eval_jacobian(
            self._model,
            self._model_state,
            J=self._jacobian_full,
            joint_S_s=self._joint_S_s,
            mask=self._eval_mask,
        )

        wp.launch(
            _site_pose_kernel,
            dim=n,
            inputs=[self._model_state.body_q, self._site_body, self._site_xform],
            outputs=[self._ee_pos, self._ee_quat],
            device=self._device,
        )
        wp.launch(
            _site_jacobian_kernel,
            dim=(n, max_dofs),
            inputs=[
                self._jacobian_full,
                self._model_state.body_q,
                self._model.body_com,
                idx.selected_articulations,
                self._site_body,
                self._site_link_row,
                self._site_xform,
                idx.controlled_dof_to_model_dof,
                idx.dof_offsets,
                idx.articulation_dof_start,
            ],
            outputs=[self._j_site],
            device=self._device,
        )
        if self._task_dim == 3:
            wp.launch(
                _copy_position_rows_kernel,
                dim=(n, max_dofs),
                inputs=[self._j_site],
                outputs=[self._j_site_pos],
                device=self._device,
            )

        self._mf_input.target_pos = inputs.target_pos
        if self._task_dim == 6:
            self._mf_input.target_quat = inputs.target_quat
        self._mf_input.bandwidth = inputs.bandwidth
        self._mf_input.solver_damping = inputs.solver_damping

        self._model_free.step(inputs=self._mf_input, outputs=outputs, dt=dt)


def _resolve_site(model: Model, site: str | list[str], selected: np.ndarray) -> tuple[list[int], list[int], list]:
    """Locate the controlled site within each selected articulation.

    The Jacobian's rows are grouped by link in articulation order, so each site
    needs both the model body it is attached to and that body's position within
    its own articulation.

    Several labels may be given, in which case each selected articulation must
    carry exactly one shape from the set. That is what lets one controller span
    robot kinds whose end-effector sites are named differently -- importers name
    sites after the asset, so the label is not always the caller's to choose.

    Args:
        model: Model to search.
        site: Shape label, or several labels to accept any one of.
        selected: Indices of the selected articulations, in controller order.

    Returns:
        Per selected articulation: the model body index, the link index within
        the articulation, and the site's transform in the body's frame.

    Raises:
        ValueError: If a label matches no shape anywhere in the model, or if a
            selected articulation does not carry exactly one matching shape.
    """
    labels = (site,) if isinstance(site, str) else tuple(site)
    if not labels:
        raise ValueError("site must name at least one shape label.")

    available = set(model.shape_label)
    unknown = [label for label in labels if label not in available]
    if unknown:
        raise ValueError(f"site label(s) {unknown} match no shape in the model; available labels: {sorted(available)}.")

    # A label matching only shapes outside the selection is allowed, so the same
    # set of names can be passed whatever subset of the fleet is controlled.
    wanted = set(labels)
    matches = [i for i, label in enumerate(model.shape_label) if label in wanted]

    shape_body = model.shape_body.numpy()
    shape_transform = model.shape_transform.numpy()
    joint_child = model.joint_child.numpy()
    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()

    # body -> (articulation, link index within it), for the selected articulations only.
    body_location: dict[int, tuple[int, int]] = {}
    for articulation in selected:
        for link, joint in enumerate(range(int(art_start[articulation]), int(art_end[articulation]))):
            body_location[int(joint_child[joint])] = (int(articulation), link)

    found: dict[int, tuple[int, int, Any]] = {}
    for shape in matches:
        body = int(shape_body[shape])
        if body not in body_location:
            continue  # belongs to an articulation we do not control
        articulation, link = body_location[body]
        if articulation in found:
            raise ValueError(
                f"articulation {articulation} carries more than one shape matching site={site!r}; "
                "each controlled articulation must have exactly one."
            )
        found[articulation] = (body, link, shape_transform[shape])

    missing = [int(a) for a in selected if int(a) not in found]
    if missing:
        raise ValueError(
            f"articulations {missing} carry no shape matching site={site!r}; every controlled "
            "articulation must have exactly one."
        )

    site_body = [found[int(a)][0] for a in selected]
    site_link = [found[int(a)][1] for a in selected]
    site_xform = [found[int(a)][2] for a in selected]
    return site_body, site_link, site_xform
