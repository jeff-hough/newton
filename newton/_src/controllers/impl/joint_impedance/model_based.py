# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedance — joint-space impedance control with
Newton model-internal dynamics.

Calls :func:`newton.eval_fk` and :func:`newton.eval_mass_matrix` on the supplied
model each step to obtain the mass matrix, then delegates all
gather/compute/scatter work to an inner
:class:`ControllerJointImpedanceModelFree` instance.

Gravity and Coriolis compensation use :func:`newton.eval_inverse_dynamics_passive`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim.articulation import eval_fk, eval_mass_matrix
from newton._src.sim.inverse_dynamics import eval_inverse_dynamics_passive
from newton._src.sim.model import Model

from ...controller import ControllerBase
from ...dof_indices import _build_dof_indices
from ...utils import _validate_articulations_are_world_rooted, _validate_with_exact_shape, _validate_with_minimum_shape
from ._common import _gather_flat_kernel, _gather_mass_matrix_blocks_kernel
from .model_free import ControllerJointImpedanceModelFree


class ControllerJointImpedance(ControllerBase):
    """Joint-space impedance controller with internally computed dynamics.

    Implements the joint-space impedance control law. This model-based variant
    computes the mass matrix, gravity, and Coriolis terms itself: it evaluates
    forward kinematics and the enabled dynamics terms from ``model`` on every
    :meth:`step`, so the caller supplies only joint positions and velocities.

    ``model`` is borrowed, not owned — it is never written to, and changes to it
    are visible to the controller immediately.

    ``model`` may contain far more than the controlled robots. Every model
    coordinate and DOF is read each step, so uncontrolled joints still
    contribute to forward kinematics — an arm mounted on an uncontrolled door
    gets the right base transform, and hence the right gravity compensation.
    Torques are produced only for the selected joints.

    .. important::
        An uncontrolled joint must still belong to an articulation to be read.
        :func:`newton.eval_fk` is evaluated per articulation, so a joint outside
        every articulation never updates its body transform and anything mounted
        on it sees a stale base. Put such a joint inside the articulation and
        leave it out of ``joints``.

    Only the **controlled** joints are restricted to 1-DOF revolute or
    prismatic, since the PD error term ``q_des - q`` is a scalar subtraction.
    Uncontrolled joints may be of any type, including free and ball joints.

    Supports heterogeneous robot fleets — selected articulations may have
    different DOF counts. Joint-space vectors are flat and ragged, so only the
    mass matrix is padded.

    See also :class:`ControllerJointImpedanceModelFree`, which takes the mass
    matrix, gravity, and Coriolis terms as inputs instead of computing them
    from a :class:`newton.Model`.

    Impedance law (terms enabled at construction):

        τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
            + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
            + [g(q)      if use_gravity_compensation  else 0]

    Args:
        model: :class:`~newton.Model` containing the robots to control. May hold
            articulations and joints that are not controlled.
        articulations: Which articulations to control, as model indices or
            labels. Labels match every articulation carrying them. ``None``
            selects all.
        joints: Which joints to control, as model indices or labels, resolved
            within the selected articulations. Labels match every joint carrying
            them, which is what makes a fleet cheap to address. ``None`` selects
            every scalar joint. Controlled DOFs are ordered by articulation then
            by model joint index, *not* by the order listed here.
        model_coord_to_sim_coord: Maps each model coordinate to its slot in
            ``inputs.joint_q``, shape ``(model.joint_coord_count,)``. ``None``
            uses the identity, i.e. the simulation arrays follow the model's own
            layout. Needed only when the controller's model is not the
            simulation's, such as when modelling error is introduced
            deliberately.
        model_dof_to_sim_dof: Maps each model DOF to its slot in
            ``inputs.joint_qd`` and ``outputs.joint_f``, shape
            ``(model.joint_dof_count,)``. Kept separate from
            ``model_coord_to_sim_coord`` because Newton indexes positions and
            velocities differently — a free joint spans 7 coordinates but 6 DOFs
            — so one array cannot serve both.
        stiffness: Position-error gain Kp, shape ``(controlled_dof_count,)``.
            Units depend on ``use_inertia_decoupling``: [1/s²] when enabled,
            since the PD term is then an acceleration premultiplied by M(q);
            otherwise [N/m or N·m/rad]. Pass an array to copy it at
            construction, or ``None`` to read ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd, [1/s] when ``use_inertia_decoupling``
            is enabled, otherwise [N·s/m or N·m·s/rad]. Same format as
            ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedance.input`.

        Dynamics fields (mass matrix, gravity, Coriolis) are computed
        internally and do not appear here.
        """

        joint_q: wp.array[wp.float32]
        """Current joint positions [m or rad], in simulation coordinate order."""
        joint_qd: wp.array[wp.float32]
        """Current joint velocities [m/s or rad/s], in simulation DOF order."""
        joint_q_des: wp.array[wp.float32]
        """Desired joint positions [m or rad], shape ``(controlled_dof_count,)``."""
        joint_qd_des: wp.array[wp.float32]
        """Desired joint velocities [m/s or rad/s], shape ``(controlled_dof_count,)``."""
        joint_qdd: wp.array[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], shape ``(controlled_dof_count,)``. ``None`` unless ``has_qdd_feedforward=True``."""
        stiffness: wp.array[wp.float32] | None
        """Position-error gain Kp, shape ``(controlled_dof_count,)``. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction."""
        damping: wp.array[wp.float32] | None
        """Velocity-error gain Kd, shape ``(controlled_dof_count,)``. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedance.output`."""

        joint_f: wp.array[wp.float32]
        """Joint torque command [N or N·m], in simulation DOF order."""

    def __init__(
        self,
        model: Model,
        *,
        articulations: list[int] | list[str] | None = None,
        joints: list[int] | list[str] | None = None,
        model_coord_to_sim_coord: wp.array[wp.int32] | None = None,
        model_dof_to_sim_dof: wp.array[wp.int32] | None = None,
        stiffness: wp.array[wp.float32] | None,
        damping: wp.array[wp.float32] | None,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
        requires_grad: bool = False,
    ):
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")

        # Resolves the selection and rejects non-scalar controlled joints.
        self._dof_indices = _build_dof_indices(model, articulations=articulations, joints=joints)
        idx = self._dof_indices

        self._device = model.device
        self._requires_grad = requires_grad
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._needs_fk = self._use_inertia or self._use_gravity or self._use_coriolis
        self._stiffness_is_live = stiffness is None
        self._damping_is_live = damping is None

        if self._needs_fk:
            _validate_articulations_are_world_rooted(model, idx.selected_articulations.numpy())

        self._model = model
        self._model_state = model.state()
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)

        if model_coord_to_sim_coord is None:
            model_coord_to_sim_coord = wp.array(
                np.arange(self._coord_count, dtype=np.int32), dtype=wp.int32, device=self._device
            )
        if model_dof_to_sim_dof is None:
            model_dof_to_sim_dof = wp.array(
                np.arange(self._dof_count, dtype=np.int32), dtype=wp.int32, device=self._device
            )

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere
        # else. Shapes derive from the finalized model and the selection.
        # ------------------------------------------------------------------
        flat_shape = (idx.controlled_dof_count,)
        for name, array, dtype, shape in (
            ("stiffness", stiffness, wp.float32, flat_shape),
            ("damping", damping, wp.float32, flat_shape),
            ("model_coord_to_sim_coord", model_coord_to_sim_coord, wp.int32, (self._coord_count,)),
            ("model_dof_to_sim_dof", model_dof_to_sim_dof, wp.int32, (self._dof_count,)),
        ):
            required = name not in ("stiffness", "damping")
            _validate_with_exact_shape(
                array=array, name=name, dtype=dtype, shape=shape, device=self._device, required=required
            )
        # ------------------------------------------------------------------

        self._model_coord_to_sim_coord = model_coord_to_sim_coord
        self._model_dof_to_sim_dof = model_dof_to_sim_dof
        self._min_len_q = int(np.max(model_coord_to_sim_coord.numpy())) + 1
        self._min_len_qd = int(np.max(model_dof_to_sim_dof.numpy())) + 1

        # Controlled DOF -> simulation slot, composed from the selection and the
        # caller's mapping. Not a constructor argument: supplying it separately
        # would be redundant and could silently disagree with the selection.
        controlled_to_sim_dof = model_dof_to_sim_dof.numpy()[idx.controlled_dof_to_model_dof.numpy()]
        controlled_dof_to_sim_dof = wp.array(controlled_to_sim_dof, dtype=wp.int32, device=self._device)
        # Derived here rather than read back off the inner controller: this class
        # builds the index array, so it already knows the output port's length.
        self._min_len_f = int(controlled_to_sim_dof.max()) + 1

        # Performance only: skipping unselected articulations changes nothing,
        # since their results are never read. mask=None is always correct.
        self._eval_mask: wp.array[wp.bool] | None = None
        if idx.robot_count < model.articulation_count:
            mask_np = np.zeros(model.articulation_count, dtype=bool)
            mask_np[idx.selected_articulations.numpy()] = True
            self._eval_mask = wp.array(mask_np, dtype=wp.bool, device=self._device)

        self._mass_matrix: wp.array3d[wp.float32] | None = None
        self._mass_matrix_full: wp.array3d[wp.float32] | None = None
        self._gravity_flat: wp.array[wp.float32] | None = None
        self._coriolis_flat: wp.array[wp.float32] | None = None

        # eval_mass_matrix writes H indexed by *model* articulation, so it is
        # allocated at full model shape and the selected blocks are extracted each
        # step. The same applies to the flat gravity/Coriolis outputs.
        if self._use_inertia:
            model_max_dofs = model.max_dofs_per_articulation
            self._mass_matrix_full = wp.zeros(
                (model.articulation_count, model_max_dofs, model_max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._mass_matrix = wp.zeros(
                (idx.robot_count, idx.max_dofs, idx.max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
        if self._use_gravity:
            self._gravity_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
        if self._use_coriolis:
            self._coriolis_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )

        dofs_per_robot = wp.array(np.diff(idx.dof_offsets.numpy()), dtype=wp.int32, device=self._device)
        self._model_free = ControllerJointImpedanceModelFree(
            dofs_per_robot=dofs_per_robot,
            stiffness=stiffness,
            damping=damping,
            use_gravity_compensation=use_gravity_compensation,
            use_coriolis_compensation=use_coriolis_compensation,
            use_inertia_decoupling=use_inertia_decoupling,
            has_qdd_feedforward=has_qdd_feedforward,
            # The inner controller reads this controller's model state, so its
            # ports are indexed in model space. Position uses coordinate indices
            # and velocity uses DOF indices; they differ once any joint upstream
            # spans more coordinates than DOFs.
            joint_q_idx=idx.controlled_dof_to_model_coord,
            joint_qd_idx=idx.controlled_dof_to_model_dof,
            gravity_force_idx=idx.controlled_dof_to_model_dof,
            coriolis_force_idx=idx.controlled_dof_to_model_dof,
            joint_f_idx=controlled_dof_to_sim_dof,
            device=self._device,
            requires_grad=requires_grad,
        )

        # Pre-wired fields forwarded to ModelFree each step.
        self._mf_input = ControllerJointImpedanceModelFree.Inputs()
        self._mf_input.joint_q = self._model_state.joint_q
        self._mf_input.joint_qd = self._model_state.joint_qd
        if self._use_inertia:
            self._mf_input.mass_matrix = self._mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat
        if self._use_coriolis:
            self._mf_input.coriolis_force = self._coriolis_flat

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
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_graphable(self) -> bool:
        return True

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` without dynamics fields."""
        d, rg = self._device, self._requires_grad
        n = self._dof_indices.controlled_dof_count
        inputs = ControllerJointImpedance.Inputs()
        inputs.joint_q = wp.zeros(self._min_len_q, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd = wp.zeros(self._min_len_qd, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_q_des = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd_des = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qdd = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if self._has_qdd else None
        inputs.stiffness = (
            wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if self._stiffness_is_live else None
        )
        inputs.damping = wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if self._damping_is_live else None
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a flat torque array."""
        outputs = ControllerJointImpedance.Outputs()
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
        """Run one impedance-control step.

        Args:
            inputs: Populated :class:`Inputs` struct. Dynamics terms are
                computed internally from the Newton model.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
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

        # Populate every model coordinate and DOF, not just the controlled ones:
        # an uncontrolled joint upstream of a controlled articulation still sets
        # its base transform, and hence its gravity torque.
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

        idx = self._dof_indices
        if self._needs_fk:
            eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        if self._use_inertia:
            eval_mass_matrix(self._model, self._model_state, H=self._mass_matrix_full, mask=self._eval_mask)
            wp.launch(
                _gather_mass_matrix_blocks_kernel,
                dim=(idx.robot_count, idx.max_dofs, idx.max_dofs),
                inputs=[
                    self._mass_matrix_full,
                    idx.selected_articulations,
                    idx.controlled_dof_to_model_dof,
                    idx.dof_offsets,
                    idx.articulation_dof_start,
                ],
                outputs=[self._mass_matrix],
                device=self._device,
            )
        if self._use_gravity or self._use_coriolis:
            eval_inverse_dynamics_passive(
                self._model,
                self._model_state,
                gravity_force=self._gravity_flat,
                coriolis_force=self._coriolis_flat,
                mask=self._eval_mask,
            )

        self._mf_input.joint_q_des = inputs.joint_q_des
        self._mf_input.joint_qd_des = inputs.joint_qd_des
        if self._has_qdd:
            self._mf_input.joint_qdd = inputs.joint_qdd
        if self._stiffness_is_live:
            self._mf_input.stiffness = inputs.stiffness
        if self._damping_is_live:
            self._mf_input.damping = inputs.damping

        self._model_free.step(inputs=self._mf_input, outputs=outputs, dt=dt)
