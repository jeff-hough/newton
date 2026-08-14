# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerDifferentialKinematics and its model-free variant."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import (
    CommandType,
    ControllerDifferentialKinematics,
    ControllerDifferentialKinematicsModelFree,
    IkMethod,
    select_joints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LINK_LEN = 0.3


def _arm(n_links=3, label="arm", site_label="tcp", with_site=True):
    """Build a planar n-link arm rotating about Y with a site at its tip."""
    builder = newton.ModelBuilder()
    prev, joints = -1, []
    for i in range(n_links):
        link = builder.add_link()
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(LINK_LEN if i else 0.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                label=f"{label}_j{i}",
            )
        )
        builder.add_shape_box(
            body=link,
            xform=wp.transform(wp.vec3(LINK_LEN * 0.5, 0.0, 0.0), wp.quat_identity()),
            hx=LINK_LEN * 0.5,
            hy=0.03,
            hz=0.03,
        )
        prev = link
    if with_site:
        builder.add_site(
            body=prev, xform=wp.transform(wp.vec3(LINK_LEN, 0.0, 0.0), wp.quat_identity()), label=site_label
        )
    builder.add_articulation(joints, label=label)
    return builder


def _fleet(*specs):
    """Finalize a scene replicating each ``(builder, count)`` spec in order."""
    scene = newton.ModelBuilder()
    for builder, count in specs:
        scene.replicate(builder, world_count=count, spacing=(2.0, 0.0, 0.0))
    return scene.finalize()


def _single(builder):
    """Finalize a scene holding one copy of a builder."""
    scene = newton.ModelBuilder()
    scene.add_builder(builder)
    return scene.finalize()


def _converge(ctrl, model, target_offsets, iterations=200, dt=0.02):
    """Drive the controller to convergence and return the final site positions.

    Args:
        ctrl: Controller under test.
        model: Model it was built from.
        target_offsets: Per-robot target, expressed relative to that robot's root
            body so a replicated fleet gets equivalent tasks.
        iterations: Number of IK steps.
        dt: Control period.
    """
    device = model.device
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    ins, outs = ctrl.input(), ctrl.output()
    ins.joint_q, ins.joint_qd = state.joint_q, state.joint_qd
    ins.target_pos = wp.array(np.asarray(target_offsets, dtype=np.float32), dtype=wp.vec3, device=device)
    if ctrl.task_dim == 6:
        ins.target_quat = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (ctrl.robot_count, 1)),
            dtype=wp.quatf,
            device=device,
        )

    for _ in range(iterations):
        ctrl.step(inputs=ins, outputs=outs, dt=dt)
        state.joint_q.assign(outs.joint_target_q.numpy())
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return ctrl._ee_pos.numpy().copy()


def _root_positions(model, ctrl):
    """Return each controlled robot's root body position at the rest pose."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    art_start = model.articulation_start.numpy()
    joint_child = model.joint_child.numpy()
    body_q = state.body_q.numpy()
    return np.array(
        [body_q[joint_child[art_start[a]]][:3] for a in ctrl._dof_indices.selected_articulations.numpy()],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Model-free
# ---------------------------------------------------------------------------


class TestDifferentialKinematicsModelFree(unittest.TestCase):
    def _make(self, dofs_list=(2,), **kwargs):
        device = wp.get_device()
        return ControllerDifferentialKinematicsModelFree(
            dofs_per_robot=wp.array(np.array(dofs_list, dtype=np.int32), dtype=wp.int32, device=device),
            **kwargs,
        )

    def test_zero_error_gives_zero_command(self):
        """Verify a site already at its target produces no joint velocity."""
        device = wp.get_device()
        ctrl = self._make(command_type=CommandType.POSITION)
        ins, outs = ctrl.input(), ctrl.output()
        ins.ee_pos = wp.array([[0.5, 0.0, 0.0]], dtype=wp.vec3, device=device)
        ins.target_pos = wp.array([[0.5, 0.0, 0.0]], dtype=wp.vec3, device=device)
        ins.jacobian = wp.array(np.ones((1, 3, 2), dtype=np.float32), dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.1)
        np.testing.assert_allclose(outs.joint_target_qd.numpy(), np.zeros(2), atol=1e-6)

    def test_transpose_method_moves_along_jacobian_transpose(self):
        """Verify TRANSPOSE yields exactly bandwidth * Jᵀe, which is checkable by hand."""
        device = wp.get_device()
        ctrl = self._make(command_type=CommandType.POSITION, ik_method=IkMethod.TRANSPOSE, bandwidth=2.0)
        ins, outs = ctrl.input(), ctrl.output()
        ins.ee_pos = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        ins.target_pos = wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        J = np.zeros((1, 3, 2), dtype=np.float32)
        J[0, 0, 0] = 1.0  # first joint moves the site along +x
        J[0, 2, 1] = 1.0  # second joint moves it along +z
        ins.jacobian = wp.array(J, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.1)
        # e = (1,0,0); Jᵀe = (1, 0); scaled by bandwidth 2.
        np.testing.assert_allclose(outs.joint_target_qd.numpy(), [2.0, 0.0], atol=1e-6)

    def test_position_target_integrates_velocity(self):
        """Verify joint_target_q is joint_q integrated forward by one period."""
        device = wp.get_device()
        ctrl = self._make(command_type=CommandType.POSITION, ik_method=IkMethod.TRANSPOSE, bandwidth=1.0)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.array([0.25, -0.5], dtype=wp.float32, device=device)
        ins.ee_pos = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        ins.target_pos = wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        J = np.zeros((1, 3, 2), dtype=np.float32)
        J[0, 0, 0] = 1.0
        ins.jacobian = wp.array(J, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.5)

        qd = outs.joint_target_qd.numpy()
        np.testing.assert_allclose(outs.joint_target_q.numpy(), [0.25, -0.5] + qd * 0.5, atol=1e-6)

    def test_heterogeneous_fleet_ignores_jacobian_padding(self):
        """Verify a short robot's padded Jacobian columns never reach the output.

        Robot 0 has 3 DOFs and robot 1 has 1, so robot 1's columns 1 and 2 are
        padding. They are filled with values that would dominate the solve if the
        solver read them.
        """
        device = wp.get_device()
        ctrl = self._make(
            dofs_list=(3, 1), command_type=CommandType.POSITION, ik_method=IkMethod.TRANSPOSE, bandwidth=1.0
        )
        self.assertEqual(ctrl.controlled_dof_count, 4)
        self.assertEqual(ctrl.max_dofs, 3)

        ins, outs = ctrl.input(), ctrl.output()
        ins.ee_pos = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)
        ins.target_pos = wp.array(
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device
        )
        J = np.zeros((2, 3, 3), dtype=np.float32)
        J[0, 0, :] = [1.0, 2.0, 3.0]
        J[1, 0, 0] = 4.0
        J[1, 0, 1:] = 1e6  # poison robot 1's padding columns
        ins.jacobian = wp.array(J, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.1)

        # Jᵀe keeps only the real columns: robot 0 -> (1,2,3), robot 1 -> (4,).
        np.testing.assert_allclose(outs.joint_target_qd.numpy(), [1.0, 2.0, 3.0, 4.0], atol=1e-4)

    def test_inputs_and_outputs_have_declared_fields(self):
        """Verify the typed structs expose exactly the documented ports."""
        ctrl = self._make(command_type=CommandType.POSE, bandwidth=None, solver_damping=None)
        ins, outs = ctrl.input(), ctrl.output()
        for field in ("joint_q", "ee_pos", "ee_quat", "target_pos", "target_quat", "jacobian"):
            self.assertIsNotNone(getattr(ins, field), f"missing input: {field}")
        self.assertIsNotNone(ins.bandwidth)  # live, because None was passed
        self.assertIsNotNone(ins.solver_damping)
        self.assertIsNotNone(outs.joint_target_q)
        self.assertIsNotNone(outs.joint_target_qd)

    def test_baked_gains_absent_from_input_struct(self):
        """Verify a gain given at construction is not exposed as a live port."""
        ctrl = self._make(bandwidth=1.5, solver_damping=0.1)
        ins = ctrl.input()
        self.assertIsNone(ins.bandwidth)
        self.assertIsNone(ins.solver_damping)

    def test_position_only_has_no_orientation_target(self):
        """Verify position control allocates no orientation target port."""
        ctrl = self._make(command_type=CommandType.POSITION)
        self.assertEqual(ctrl.task_dim, 3)
        self.assertIsNone(ctrl.input().target_quat)

    def test_is_graphable_only_false_for_svd(self):
        """Verify every method except SVD reports itself capturable."""
        for method in IkMethod:
            ctrl = self._make(ik_method=method)
            self.assertEqual(ctrl.is_graphable(), method != IkMethod.SVD, f"{method.name}")


class TestDifferentialKinematicsModelFreeErrors(unittest.TestCase):
    def _kwargs(self, device, **overrides):
        base = {
            "dofs_per_robot": wp.array(np.array([2], dtype=np.int32), dtype=wp.int32, device=device),
        }
        base.update(overrides)
        return base

    def test_rejects_orientation_weight_for_position_control(self):
        """Verify orientation weighting is rejected when there are no orientation rows."""
        device = wp.get_device()
        with self.assertRaisesRegex(ValueError, "requires CommandType.POSE"):
            ControllerDifferentialKinematicsModelFree(
                **self._kwargs(device, command_type=CommandType.POSITION, orientation_weight=0.5)
            )

    def test_rejects_avoidance_without_limits(self):
        """Verify enabling null-space avoidance without joint limits is rejected."""
        device = wp.get_device()
        with self.assertRaisesRegex(ValueError, "joint_pos_lower is required"):
            ControllerDifferentialKinematicsModelFree(**self._kwargs(device, joint_limit_avoidance_gain=1.0))

    def test_rejects_duplicate_output_indices(self):
        """Verify two controlled DOFs cannot write the same output slot."""
        device = wp.get_device()
        with self.assertRaisesRegex(ValueError, "must be unique"):
            ControllerDifferentialKinematicsModelFree(
                **self._kwargs(device, joint_target_qd_idx=wp.array([0, 0], dtype=wp.int32, device=device))
            )

    def test_rejects_wrong_jacobian_shape(self):
        """Verify a Jacobian whose shape disagrees with the fleet is rejected."""
        device = wp.get_device()
        ctrl = ControllerDifferentialKinematicsModelFree(**self._kwargs(device, command_type=CommandType.POSITION))
        ins, outs = ctrl.input(), ctrl.output()
        ins.jacobian = wp.zeros((1, 3, 5), dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.1)

    def test_rejects_bad_ik_method_type(self):
        """Verify a non-enum ik_method is rejected rather than silently mishandled."""
        device = wp.get_device()
        with self.assertRaises(TypeError):
            ControllerDifferentialKinematicsModelFree(**self._kwargs(device, ik_method="dls"))


# ---------------------------------------------------------------------------
# Model-based
# ---------------------------------------------------------------------------


class TestDifferentialKinematics(unittest.TestCase):
    def test_converges_to_a_reachable_target(self):
        """Verify the site reaches a target well inside the workspace."""
        model = _single(_arm())
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSITION)
        target = _root_positions(model, ctrl) + np.array([0.4, 0.0, 0.35], dtype=np.float32)

        final = _converge(ctrl, model, target)
        np.testing.assert_allclose(final, target, atol=0.05)

    def test_replicated_fleet_solves_each_robot_identically(self):
        """Verify replicas given equivalent tasks produce equivalent solutions.

        The robots sit at different world positions, so the targets are offset
        from each root; a per-robot indexing error would break the symmetry.
        """
        model = _fleet((_arm(), 3))
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSITION)
        self.assertEqual(ctrl.robot_count, 3)
        target = _root_positions(model, ctrl) + np.array([0.4, 0.0, 0.35], dtype=np.float32)

        final = _converge(ctrl, model, target)
        np.testing.assert_allclose(final, target, atol=0.05)
        # Identical tasks must give identical joint solutions across replicas.
        errors = np.linalg.norm(final - target, axis=1)
        np.testing.assert_allclose(errors, errors[0], atol=1e-5)

    def test_pose_control_tracks_orientation(self):
        """Verify full-pose control drives the site orientation toward the target."""
        model = _single(_arm())
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSE)
        self.assertEqual(ctrl.task_dim, 6)
        target = _root_positions(model, ctrl) + np.array([0.4, 0.0, 0.35], dtype=np.float32)

        _converge(ctrl, model, target)
        # Identity target orientation: the site frame should end up near identity.
        quat = ctrl._ee_quat.numpy()[0]
        self.assertGreater(abs(float(quat[3])), 0.9, f"orientation not tracked: {quat}")

    def test_site_jacobian_predicts_site_motion(self):
        """Verify the site Jacobian matches finite differences of the site pose.

        This is what pins the COM-to-site shift: ``eval_jacobian`` returns the
        velocity of the link's centre of mass, and the site sits a link-half away,
        so omitting the ``omega x offset`` term leaves a Jacobian that is still a
        descent direction — convergence tests cannot see the error, but a
        first-order prediction of the site's motion can.
        """
        model = _single(_arm())
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSITION)
        device = model.device
        n_dofs = ctrl.controlled_dof_count

        state = model.state()
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_qd = state.joint_qd
        ins.target_pos = wp.zeros(1, dtype=wp.vec3, device=device)

        def site_pos_and_jacobian(q):
            ins.joint_q = wp.array(np.asarray(q, dtype=np.float32), dtype=wp.float32, device=device)
            ctrl.step(inputs=ins, outputs=outs, dt=0.0)
            return ctrl._ee_pos.numpy()[0].copy(), ctrl._j_site.numpy()[0][:3].copy()

        # A bent pose, so the arm is away from its singular straight configuration.
        q0 = np.array([0.3, -0.5, 0.7], dtype=np.float32)
        pos0, J = site_pos_and_jacobian(q0)

        eps = 1e-4
        for k in range(n_dofs):
            dq = np.zeros(n_dofs, dtype=np.float32)
            dq[k] = eps
            pos1, _ = site_pos_and_jacobian(q0 + dq)
            measured = (pos1 - pos0) / eps
            np.testing.assert_allclose(
                measured,
                J[:, k],
                atol=2e-3,
                err_msg=f"site Jacobian column {k} disagrees with finite differences",
            )

    def test_output_is_sized_for_the_scattered_slots(self):
        """Verify output() allocates for the simulation slots, not the controlled count."""
        model = _fleet((_arm(), 3))
        ctrl = ControllerDifferentialKinematics(model, site="tcp", articulations=[1], command_type=CommandType.POSITION)
        self.assertEqual(ctrl.controlled_dof_count, 3)  # robot 1 owns model DOFs 3..5
        outs = ctrl.output()
        self.assertEqual(outs.joint_target_q.shape, (6,))
        self.assertEqual(outs.joint_target_qd.shape, (6,))

    def test_gain_liveness_matches_the_input_struct(self):
        """Verify a gain is exposed as a live port only when it was not baked."""
        model = _single(_arm())
        baked = ControllerDifferentialKinematics(
            model, site="tcp", command_type=CommandType.POSITION, bandwidth=2.0, solver_damping=0.1
        )
        self.assertIsNone(baked.input().bandwidth)
        self.assertIsNone(baked.input().solver_damping)

        live = ControllerDifferentialKinematics(
            model, site="tcp", command_type=CommandType.POSITION, bandwidth=None, solver_damping=None
        )
        self.assertEqual(live.input().bandwidth.shape, (live.robot_count,))
        self.assertEqual(live.input().solver_damping.shape, (live.robot_count,))

    def test_position_target_output_follows_the_coordinate_layout(self):
        """Verify joint_target_q is sized for the simulation's target layout.

        Under the coordinate layout that port is indexed by joint coordinate, not
        by DOF, and behind a free joint the two differ — the arm sits at
        coordinates 7 and 8 but DOFs 6 and 7.
        """
        previous = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        try:
            builder = newton.ModelBuilder()
            torso = builder.add_link()
            j_free = builder.add_joint_free(child=torso)
            builder.add_shape_box(body=torso, hx=0.1, hy=0.1, hz=0.1)
            prev, joints = torso, [j_free]
            for i in range(2):
                link = builder.add_link()
                joints.append(
                    builder.add_joint_revolute(
                        parent=prev,
                        child=link,
                        axis=wp.vec3(0.0, 1.0, 0.0),
                        parent_xform=wp.transform(wp.vec3(LINK_LEN if i else 0.0, 0.0, 0.0), wp.quat_identity()),
                        child_xform=wp.transform_identity(),
                        label=f"arm_j{i}",
                    )
                )
                builder.add_shape_box(body=link, hx=LINK_LEN * 0.5, hy=0.03, hz=0.03)
                prev = link
            builder.add_site(
                body=prev, xform=wp.transform(wp.vec3(LINK_LEN, 0.0, 0.0), wp.quat_identity()), label="tcp"
            )
            builder.add_articulation(joints, label="mobile")
            model = _single(builder)
            self.assertTrue(model.use_coord_layout_targets)
            self.assertEqual(model.joint_coord_count, 9)
            self.assertEqual(model.joint_dof_count, 8)

            ctrl = ControllerDifferentialKinematics(
                model, site="tcp", joints=["arm_j0", "arm_j1"], command_type=CommandType.POSITION
            )
            outs = ctrl.output()
            # Coordinates 7 and 8 for positions; DOFs 6 and 7 for velocities.
            self.assertEqual(outs.joint_target_q.shape, (9,))
            self.assertEqual(outs.joint_target_qd.shape, (8,))
        finally:
            newton.use_coord_layout_targets = previous

    def test_selection_controls_a_subset_of_articulations(self):
        """Verify unselected robots receive no target at all."""
        model = _fleet((_arm(), 3))
        ctrl = ControllerDifferentialKinematics(model, site="tcp", articulations=[1], command_type=CommandType.POSITION)
        self.assertEqual(ctrl.robot_count, 1)
        self.assertEqual(ctrl.controlled_joints, (3, 4, 5))

        device = model.device
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q, ins.joint_qd = state.joint_q, state.joint_qd
        ins.target_pos = wp.array(
            _root_positions(model, ctrl) + np.array([0.4, 0.0, 0.35], dtype=np.float32),
            dtype=wp.vec3,
            device=device,
        )
        outs.joint_target_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.02)

        qd = outs.joint_target_qd.numpy()
        self.assertTrue(np.any(qd[3:6] != 0.0), "selected robot received no command")
        np.testing.assert_allclose(qd[0:3], 0.0, atol=1e-9)
        np.testing.assert_allclose(qd[6:9], 0.0, atol=1e-9)

    def test_select_joints_excludes_a_joint_from_a_controlled_robot(self):
        """Verify a joint excluded via select_joints is never commanded."""
        model = _single(_arm(n_links=3))
        controlled = select_joints(model, exclude_joints=["arm_j2"])
        ctrl = ControllerDifferentialKinematics(model, site="tcp", joints=controlled, command_type=CommandType.POSITION)
        self.assertEqual(ctrl.controlled_joints, (0, 1))

        device = model.device
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q, ins.joint_qd = state.joint_q, state.joint_qd
        ins.target_pos = wp.array(
            _root_positions(model, ctrl) + np.array([0.4, 0.0, 0.35], dtype=np.float32),
            dtype=wp.vec3,
            device=device,
        )
        outs.joint_target_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.02)

        qd = outs.joint_target_qd.numpy()
        self.assertTrue(np.any(qd[:2] != 0.0))
        self.assertEqual(float(qd[2]), 0.0, "excluded joint was commanded")

    def test_heterogeneous_fleet(self):
        """Verify articulations with different DOF counts are controlled together."""
        model = _fleet((_arm(n_links=3, label="long"), 1), (_arm(n_links=2, label="short"), 1))
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSITION)
        self.assertEqual(ctrl.robot_count, 2)
        self.assertEqual(ctrl.controlled_dof_count, 5)
        self.assertEqual(ctrl.max_dofs, 3)

        target = _root_positions(model, ctrl) + np.array([0.35, 0.0, 0.25], dtype=np.float32)
        final = _converge(ctrl, model, target)
        np.testing.assert_allclose(final, target, atol=0.08)

    def test_floating_base_coordinate_dof_divergence(self):
        """Verify a free joint upstream does not corrupt the joint-space indexing.

        Behind a free joint the arm's coordinate and DOF indices differ, so a
        controller that conflated them would read a quaternion component as a
        joint angle.
        """
        builder = newton.ModelBuilder()
        torso = builder.add_link()
        j_free = builder.add_joint_free(child=torso)
        builder.add_shape_box(body=torso, hx=0.1, hy=0.1, hz=0.1)
        prev, joints = torso, [j_free]
        for i in range(2):
            link = builder.add_link()
            joints.append(
                builder.add_joint_revolute(
                    parent=prev,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(LINK_LEN if i else 0.0, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    label=f"arm_j{i}",
                )
            )
            builder.add_shape_box(
                body=link,
                xform=wp.transform(wp.vec3(LINK_LEN * 0.5, 0.0, 0.0), wp.quat_identity()),
                hx=LINK_LEN * 0.5,
                hy=0.03,
                hz=0.03,
            )
            prev = link
        builder.add_site(body=prev, xform=wp.transform(wp.vec3(LINK_LEN, 0.0, 0.0), wp.quat_identity()), label="tcp")
        builder.add_articulation(joints, label="mobile")
        model = _single(builder)
        self.assertNotEqual(model.joint_coord_count, model.joint_dof_count)

        ctrl = ControllerDifferentialKinematics(
            model, site="tcp", joints=["arm_j0", "arm_j1"], command_type=CommandType.POSITION
        )
        self.assertEqual(ctrl.controlled_dof_count, 2)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q, ins.joint_qd = state.joint_q, state.joint_qd
        ins.target_pos = wp.array(np.array([[0.4, 0.0, 0.3]], dtype=np.float32), dtype=wp.vec3, device=model.device)
        outs.joint_target_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.02)

        qd = outs.joint_target_qd.numpy()
        # The free joint owns DOFs 0..5 and must never be commanded.
        np.testing.assert_allclose(qd[:6], 0.0, atol=1e-9)
        self.assertTrue(np.any(qd[6:8] != 0.0), "arm received no command")

    def test_solver_methods_all_produce_finite_commands(self):
        """Verify every graph-capturable IK method runs and stays finite."""
        model = _single(_arm())
        target = np.array([[0.4, 0.0, 0.35]], dtype=np.float32)
        for method in IkMethod:
            if method == IkMethod.SVD:
                continue  # covered separately; needs torch
            with self.subTest(method=method.name):
                ctrl = ControllerDifferentialKinematics(
                    model, site="tcp", command_type=CommandType.POSITION, ik_method=method
                )
                final = _converge(ctrl, model, _root_positions(model, ctrl) + target, iterations=50)
                self.assertTrue(np.all(np.isfinite(final)), f"{method.name} produced non-finite output")

    def test_svd_method_matches_damped_least_squares(self):
        """Verify the torch-backed SVD solve agrees with the Warp DLS solve."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        model = _single(_arm())
        offset = np.array([[0.4, 0.0, 0.35]], dtype=np.float32)
        results = {}
        for method in (IkMethod.DAMPED_LEAST_SQUARES, IkMethod.SVD):
            ctrl = ControllerDifferentialKinematics(
                model, site="tcp", command_type=CommandType.POSITION, ik_method=method
            )
            results[method] = _converge(ctrl, model, _root_positions(model, ctrl) + offset)
        np.testing.assert_allclose(results[IkMethod.SVD], results[IkMethod.DAMPED_LEAST_SQUARES], atol=0.05)

    def test_null_space_avoidance_uses_model_joint_limits(self):
        """Verify joint limits default to the model's own at the controlled DOFs."""
        model = _single(_arm())
        ctrl = ControllerDifferentialKinematics(
            model, site="tcp", command_type=CommandType.POSITION, joint_limit_avoidance_gain=1.0
        )
        lower = ctrl._model_free._joint_pos_lower.numpy()
        upper = ctrl._model_free._joint_pos_upper.numpy()
        self.assertEqual(lower.shape, (ctrl.controlled_dof_count,))
        np.testing.assert_allclose(lower, model.joint_limit_lower.numpy()[list(range(3))])
        np.testing.assert_allclose(upper, model.joint_limit_upper.numpy()[list(range(3))])

    def test_is_graphable_and_capture(self):
        """Verify a capturable configuration can actually be recorded in a CUDA graph."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("graph capture needs a CUDA device")
        model = _single(_arm())
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSITION)
        self.assertTrue(ctrl.is_graphable())

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q, ins.joint_qd = state.joint_q, state.joint_qd
        dt = wp.array([0.02], dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=dt)  # warm up module loads
        with wp.ScopedCapture() as capture:
            ctrl.step(inputs=ins, outputs=outs, dt=dt)
        wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(outs.joint_target_qd.numpy())))


class TestDifferentialKinematicsErrors(unittest.TestCase):
    def test_rejects_unknown_site_label(self):
        """Verify a mistyped site label fails loudly at construction."""
        model = _single(_arm())
        with self.assertRaisesRegex(ValueError, "match no shape in the model"):
            ControllerDifferentialKinematics(model, site="gripper")

    def test_site_accepts_several_labels_for_a_mixed_fleet(self):
        """Verify one controller can span robot kinds whose sites are named differently.

        Importers label sites after the asset, so two vendors' robots may carry
        `tcp_a` and `tcp_b`. Each articulation must still hold exactly one.
        """
        model = _fleet(
            (_arm(n_links=3, label="kind_a", site_label="tcp_a"), 1),
            (_arm(n_links=2, label="kind_b", site_label="tcp_b"), 1),
        )
        ctrl = ControllerDifferentialKinematics(model, site=["tcp_a", "tcp_b"], command_type=CommandType.POSITION)
        self.assertEqual(ctrl.robot_count, 2)
        self.assertEqual(ctrl.controlled_dof_count, 5)  # 3 + 2, ragged

        target = _root_positions(model, ctrl) + np.array([0.3, 0.0, 0.25], dtype=np.float32)
        final = _converge(ctrl, model, target)
        np.testing.assert_allclose(final, target, atol=0.08)

    def test_site_label_outside_the_selection_is_allowed(self):
        """Verify an unused label does not block a narrowed selection.

        The same set of names should be passable whatever subset of the fleet is
        being controlled.
        """
        model = _fleet((_arm(label="kind_a", site_label="tcp_a"), 1), (_arm(label="kind_b", site_label="tcp_b"), 1))
        ctrl = ControllerDifferentialKinematics(
            model, site=["tcp_a", "tcp_b"], articulations=[0], command_type=CommandType.POSITION
        )
        self.assertEqual(ctrl.robot_count, 1)

    def test_rejects_site_label_matching_nothing_in_the_model(self):
        """Verify a mistyped label in the set is caught rather than silently ignored."""
        model = _fleet((_arm(site_label="tcp_a"), 2))
        with self.assertRaisesRegex(ValueError, "match no shape in the model"):
            ControllerDifferentialKinematics(model, site=["tcp_a", "tpc_a"])

    def test_rejects_articulation_matching_several_site_labels(self):
        """Verify a robot carrying two of the accepted labels is ambiguous, not arbitrary."""
        builder = _arm(site_label="tcp_a")
        builder.add_site(body=2, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), label="tcp_b")
        model = _single(builder)
        with self.assertRaisesRegex(ValueError, "more than one shape matching"):
            ControllerDifferentialKinematics(model, site=["tcp_a", "tcp_b"])

    def test_rejects_empty_site_list(self):
        """Verify an empty label list is rejected rather than read as select-nothing."""
        model = _single(_arm())
        with self.assertRaisesRegex(ValueError, "at least one shape label"):
            ControllerDifferentialKinematics(model, site=[])

    def test_rejects_articulation_without_a_site(self):
        """Verify every controlled articulation must carry the site."""
        model = _fleet((_arm(label="with_site"), 1), (_arm(label="no_site", with_site=False), 1))
        with self.assertRaisesRegex(ValueError, "carry no shape matching"):
            ControllerDifferentialKinematics(model, site="tcp")

    def test_rejects_duplicate_site_in_one_articulation(self):
        """Verify an articulation carrying two sites with the same label is rejected."""
        builder = _arm()
        builder.add_site(body=2, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), label="tcp")
        model = _single(builder)
        with self.assertRaisesRegex(ValueError, "more than one shape matching"):
            ControllerDifferentialKinematics(model, site="tcp")

    def test_rejects_non_scalar_controlled_joint(self):
        """Verify a free joint cannot be selected for control."""
        builder = newton.ModelBuilder()
        torso = builder.add_link()
        j_free = builder.add_joint_free(child=torso)
        builder.add_shape_box(body=torso, hx=0.1, hy=0.1, hz=0.1)
        builder.add_site(body=torso, xform=wp.transform_identity(), label="tcp")
        builder.add_articulation([j_free], label="floater")
        model = _single(builder)
        with self.assertRaisesRegex(ValueError, "1-DOF revolute or prismatic"):
            ControllerDifferentialKinematics(model, site="tcp")

    def test_rejects_externally_mounted_articulation(self):
        """Verify the articulation-rooting check applies here too.

        This controller always evaluates forward kinematics and the Jacobian, so
        it is exposed to the same uninitialised-scratch hazard as the impedance
        controller.
        """
        builder = newton.ModelBuilder()
        mount = builder.add_link()
        builder.add_joint_revolute(parent=-1, child=mount, axis=wp.vec3(0.0, 1.0, 0.0))
        builder.add_shape_box(body=mount, hx=0.1, hy=0.02, hz=0.02)
        arm = builder.add_link()
        j_arm = builder.add_joint_revolute(
            parent=mount,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_box(body=arm, hx=0.15, hy=0.02, hz=0.02)
        builder.add_site(body=arm, xform=wp.transform_identity(), label="tcp")
        builder.add_articulation([j_arm], label="arm")
        model = _single(builder)
        with self.assertRaisesRegex(ValueError, "rooted at the world"):
            ControllerDifferentialKinematics(model, site="tcp")

    def test_rejects_wrong_mapping_shape(self):
        """Verify each model-to-simulation mapping must match the model."""
        model = _single(_arm())
        device = model.device
        bad = wp.array(np.arange(99, dtype=np.int32), dtype=wp.int32, device=device)
        for name in ("model_coord_to_sim_coord", "model_dof_to_sim_dof"):
            with self.subTest(argument=name), self.assertRaises(ValueError):
                ControllerDifferentialKinematics(model, site="tcp", **{name: bad})

    def test_rejects_short_joint_q(self):
        """Verify a joint_q shorter than the model is rejected before the gather."""
        model = _single(_arm())
        ctrl = ControllerDifferentialKinematics(model, site="tcp", command_type=CommandType.POSITION)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(1, dtype=wp.float32, device=model.device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.02)


if __name__ == "__main__":
    unittest.main()
