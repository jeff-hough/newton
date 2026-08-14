# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerJointImpedance and ControllerJointImpedanceModelFree."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import ControllerJointImpedance, ControllerJointImpedanceModelFree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iota(n, device):
    """Return a wp.array[int32] of [0, 1, …, n-1]."""
    return wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device)


def _dofs_arr(dofs_list, device):
    """Return a wp.array[int32] from a list of per-robot DOF counts."""
    return wp.array(np.array(dofs_list, dtype=np.int32), device=device)


def _gains(controlled_dof_count, value, device):
    """Return a (controlled_dof_count,) float32 gain array filled with value."""
    return wp.full(controlled_dof_count, value, dtype=wp.float32, device=device)


def _flat(data, device):
    """Return a flat float32 Warp array from any array-like."""
    return wp.array(np.array(data, dtype=np.float32).flatten(), dtype=wp.float32, device=device)


def _build_single_prismatic():
    """Build a finalized one-robot, one-DOF prismatic-joint model."""
    builder = newton.ModelBuilder()
    link = builder.add_link()
    j = builder.add_joint_prismatic(
        parent=-1,
        child=link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j], label="robot")
    return builder.finalize()


def _build_two_robot_mixed():
    """Build a finalized model with robot 0 (2 revolute DOFs) and robot 1 (1 prismatic DOF)."""
    builder = newton.ModelBuilder()
    # Robot 0: 2-DOF revolute chain
    l0a = builder.add_link()
    l0b = builder.add_link()
    j0a = builder.add_joint_revolute(
        parent=-1,
        child=l0a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j0b = builder.add_joint_revolute(
        parent=l0a,
        child=l0b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j0a, j0b], label="robot0")
    # Robot 1: 1-DOF prismatic
    l1 = builder.add_link()
    j1 = builder.add_joint_prismatic(
        parent=-1,
        child=l1,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j1], label="robot1")
    return builder.finalize()


def _build_two_link_arm_with_mass():
    """Build a finalized single-articulation 2-DOF arm whose links carry inertia."""
    builder = newton.ModelBuilder()
    joints, prev = [], -1
    for i in range(2):
        link = builder.add_link()
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2 if i else 0.0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
        builder.add_shape_box(
            body=link,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
            hx=0.02,
            hy=0.02,
            hz=0.1,
        )
        prev = link
    builder.add_articulation(joints, label="arm")
    return builder.finalize()


def _build_two_arms_with_mass():
    """Build two 2-DOF articulations with different link sizes, so their mass matrices differ.

    Articulation 1 starts at model DOF 2, so selecting it exercises both the
    articulation block index and the per-articulation DOF base offset — a
    single-articulation model leaves both at zero and cannot detect an error.
    """
    builder = newton.ModelBuilder()
    for arm, half_len in enumerate((0.1, 0.25)):
        joints, prev = [], -1
        for i in range(2):
            link = builder.add_link()
            joints.append(
                builder.add_joint_revolute(
                    parent=prev,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(
                        wp.vec3(float(arm), 0.0, -2.0 * half_len if i else 0.0), wp.quat_identity()
                    ),
                    child_xform=wp.transform_identity(),
                )
            )
            builder.add_shape_box(
                body=link,
                xform=wp.transform(wp.vec3(0.0, 0.0, -half_len), wp.quat_identity()),
                hx=0.02,
                hy=0.02,
                hz=half_len,
            )
            prev = link
        builder.add_articulation(joints, label=f"arm_{arm}")
    return builder.finalize()


def _build_arm_on_external_mount(*, mount_fixed):
    """Build an arm whose root joint hangs off a body driven outside its articulation.

    Args:
        mount_fixed: Whether the mounting joint is fixed (zero DOF) rather than
            revolute. Both corrupt inverse dynamics — the mount's body is never
            written by the arm's traversal regardless of how many DOFs it has.
    """
    builder = newton.ModelBuilder()
    mount = builder.add_link()
    if mount_fixed:
        builder.add_joint_fixed(
            parent=-1, child=mount, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )
    else:
        builder.add_joint_revolute(
            parent=-1,
            child=mount,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
    builder.add_shape_box(body=mount, hx=0.1, hy=0.02, hz=0.02)
    arm = builder.add_link()
    j_arm = builder.add_joint_revolute(
        parent=mount,  # parent body is driven by a joint outside this articulation
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        label="arm",
    )
    builder.add_shape_box(
        body=arm, xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()), hx=0.15, hy=0.02, hz=0.02
    )
    builder.add_articulation([j_arm], label="arm")
    return builder.finalize()


def _build_arm_on_door(*, door_in_articulation):
    """Build a 1-DOF arm mounted on a revolute "door" that tilts about Y.

    Args:
        door_in_articulation: Whether the door joint belongs to the articulation.
            Only then does ``eval_fk`` update the door's body transform, and only
            then does its position reach the arm's gravity torque.
    """
    builder = newton.ModelBuilder()
    door = builder.add_link()
    j_door = builder.add_joint_revolute(
        parent=-1,
        child=door,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        label="door",
    )
    builder.add_shape_box(
        body=door, xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()), hx=0.25, hy=0.02, hz=0.02
    )
    arm = builder.add_link()
    j_arm = builder.add_joint_revolute(
        parent=door,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        label="arm",
    )
    builder.add_shape_box(
        body=arm, xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()), hx=0.15, hy=0.02, hz=0.02
    )
    builder.add_articulation([j_door, j_arm] if door_in_articulation else [j_arm], label="rig")
    return builder.finalize()


def _make_mf(
    *,
    dofs_list,
    kp,
    kd,
    device,
    use_gravity=False,
    use_coriolis=False,
    use_inertia=False,
    has_qdd=False,
):
    """Construct a ControllerJointImpedanceModelFree with identity indices."""
    total_dofs = sum(dofs_list)
    return ControllerJointImpedanceModelFree(
        dofs_per_robot=_dofs_arr(dofs_list, device),
        stiffness=_gains(total_dofs, kp, device),
        damping=_gains(total_dofs, kd, device),
        use_gravity_compensation=use_gravity,
        use_coriolis_compensation=use_coriolis,
        use_inertia_decoupling=use_inertia,
        has_qdd_feedforward=has_qdd,
        device=device,
    )


def _run_mf(ctrl, *, q, qd, q_des, qd_des, device, **extras):
    """Run one step on a ModelFree controller and return the torque array."""
    ins = ctrl.input()
    ins.joint_q = _flat(q, device)
    ins.joint_qd = _flat(qd, device)
    ins.joint_q_des = _flat(q_des, device)
    ins.joint_qd_des = _flat(qd_des, device)
    for k, v in extras.items():
        setattr(ins, k, v)
    outs = ctrl.output()
    ctrl.step(inputs=ins, outputs=outs, dt=0.01)
    return outs.joint_f.numpy()


# ---------------------------------------------------------------------------
# ControllerJointImpedanceModelFree — homogeneous
# ---------------------------------------------------------------------------


class TestControllerJointImpedanceModelFree(unittest.TestCase):
    def test_zero_error_gives_zero_torque(self):
        """Verify that zero position and velocity error produces zero torque."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3], kp=10.0, kd=1.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.1, 0.2, 0.3], qd=[0.0, 0.0, 0.0], q_des=[0.1, 0.2, 0.3], qd_des=[0.0, 0.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, np.zeros(3, dtype=np.float32), atol=1e-5)

    def test_position_error_produces_stiffness_torque(self):
        """Verify τ = Kp * (q_des - q) when Kd=0."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3], kp=5.0, kd=0.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0, 0.0], qd=[0.0, 0.0, 0.0], q_des=[1.0, 0.0, 0.0], qd_des=[0.0, 0.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, [5.0, 0.0, 0.0], atol=1e-5)

    def test_velocity_error_produces_damping_torque(self):
        """Verify τ = Kd * (qd_des - qd) when Kp=0."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[3], kp=0.0, kd=2.0, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0, 0.0], qd=[0.0, 0.0, 0.0], q_des=[0.0, 0.0, 0.0], qd_des=[0.0, 1.0, 0.0], device=device
        )
        np.testing.assert_allclose(tau, [0.0, 2.0, 0.0], atol=1e-5)

    def test_multiple_robots_independent(self):
        """Verify that torques for each robot depend only on that robot's error."""
        device = wp.get_device()
        robot_count, num_dofs = 3, 2
        ctrl = _make_mf(dofs_list=[num_dofs] * robot_count, kp=1.0, kd=0.0, device=device)
        q_des = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        q = np.zeros((robot_count, num_dofs), dtype=np.float32)
        tau = _run_mf(ctrl, q=q, qd=q * 0, q_des=q_des, qd_des=q * 0, device=device)
        np.testing.assert_allclose(tau, q_des.flatten(), atol=1e-5)

    def test_inertia_decoupling_scales_by_mass_matrix(self):
        """Verify τ = M @ (Kp * Δq) when use_inertia_decoupling=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device, use_inertia=True)
        M = wp.array(np.eye(2, dtype=np.float32).reshape(1, 2, 2) * 2.0, dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[1.0, 1.0], qd_des=[0.0, 0.0], device=device, mass_matrix=M
        )
        np.testing.assert_allclose(tau, [2.0, 2.0], atol=1e-5)

    def test_gravity_compensation_adds_to_tau(self):
        """Verify gravity_force is added to τ when use_gravity_compensation=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=0.0, kd=0.0, device=device, use_gravity=True)
        grav = wp.array([3.0, 4.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[0.0, 0.0], qd_des=[0.0, 0.0], device=device, gravity_force=grav
        )
        np.testing.assert_allclose(tau, [3.0, 4.0], atol=1e-5)

    def test_coriolis_compensation_adds_to_tau(self):
        """Verify coriolis_force is added to τ when use_coriolis_compensation=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=0.0, kd=0.0, device=device, use_coriolis=True)
        cor = wp.array([1.0, -1.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl, q=[0.0, 0.0], qd=[0.0, 0.0], q_des=[0.0, 0.0], qd_des=[0.0, 0.0], device=device, coriolis_force=cor
        )
        np.testing.assert_allclose(tau, [1.0, -1.0], atol=1e-5)

    def test_qdd_feedforward_adds_before_inertia(self):
        """Verify qdd feedforward is included inside M @ (PD + qdd) when use_inertia=True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=0.0, kd=0.0, device=device, use_inertia=True, has_qdd=True)
        M = wp.array(np.eye(2, dtype=np.float32).reshape(1, 2, 2) * 3.0, dtype=wp.float32, device=device)
        qdd = wp.array([1.0, 0.0], dtype=wp.float32, device=device)
        tau = _run_mf(
            ctrl,
            q=[0.0, 0.0],
            qd=[0.0, 0.0],
            q_des=[0.0, 0.0],
            qd_des=[0.0, 0.0],
            device=device,
            mass_matrix=M,
            joint_qdd=qdd,
        )
        np.testing.assert_allclose(tau, [3.0, 0.0], atol=1e-5)

    def test_live_stiffness_port(self):
        """Verify stiffness supplied via inputs.stiffness each step is applied correctly."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr([2], device),
            stiffness=None,
            damping=wp.zeros(2, dtype=wp.float32, device=device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([2.0, 0.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(2, dtype=wp.float32, device=device)
        ins.stiffness = wp.array([3.0, 3.0], dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        np.testing.assert_allclose(outs.joint_f.numpy(), [6.0, 0.0], atol=1e-5)

    def test_is_graphable(self):
        """Verify the controller reports is_graphable() == True."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        self.assertTrue(ctrl.is_graphable())

    def test_inputs_has_required_fields(self):
        """Verify input() returns a namespace with all declared port fields present."""
        device = wp.get_device()
        ctrl = _make_mf(
            dofs_list=[3],
            kp=1.0,
            kd=0.0,
            device=device,
            use_gravity=True,
            use_coriolis=True,
            use_inertia=True,
            has_qdd=True,
        )
        ins = ctrl.input()
        for field in (
            "joint_q",
            "joint_qd",
            "joint_q_des",
            "joint_qd_des",
            "joint_qdd",
            "mass_matrix",
            "gravity_force",
            "coriolis_force",
        ):
            self.assertTrue(hasattr(ins, field), f"Missing field: {field}")

    def test_outputs_has_joint_f(self):
        """Verify output() returns a flat array of size sum(dofs_per_robot)."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        outs = ctrl.output()
        self.assertTrue(hasattr(outs, "joint_f"))
        self.assertEqual(outs.joint_f.shape, (2,))

    def test_output_has_joint_f(self):
        """Verify output() always returns a struct with joint_f."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr([2], device),
            stiffness=wp.ones(2, dtype=wp.float32, device=device),
            damping=wp.zeros(2, dtype=wp.float32, device=device),
            device=device,
        )
        outs = ctrl.output()
        self.assertTrue(hasattr(outs, "joint_f"))

    def test_partial_sim_indices(self):
        """Verify gather/scatter correctly selects a controller-DOF subset from a larger sim array."""
        device = wp.get_device()
        indices = wp.array([1, 3], dtype=wp.int32, device=device)
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr([2], device),
            joint_q_idx=indices,
            joint_qd_idx=indices,
            joint_f_idx=indices,
            stiffness=wp.ones(2, dtype=wp.float32, device=device),
            damping=wp.zeros(2, dtype=wp.float32, device=device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(4, dtype=wp.float32, device=device)
        # Targets are controlled-order: one entry per controlled DOF, not per sim slot.
        ins.joint_q_des = wp.array([5.0, 3.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(2, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        result = outs.joint_f.numpy()
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 5.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)
        self.assertAlmostEqual(result[3], 3.0, places=5)

    def test_duplicate_output_indices_raises(self):
        """Verify that overlapping scatter indices raise ValueError at construction."""
        device = wp.get_device()
        # Two robots, both claiming DOF slot 0 as their output — undefined scatter behaviour.
        duplicate_indices = wp.array([0, 0], dtype=wp.int32, device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                dofs_per_robot=_dofs_arr([1, 1], device),
                joint_f_idx=duplicate_indices,
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_2d_dofs_per_robot_raises(self):
        """Verify a 2-D dofs_per_robot raises instead of silently deriving incorrect robot_count."""
        device = wp.get_device()
        # A (2, 3) array has size 6, which would otherwise be read as 6 robots.
        dofs_2d = wp.full((2, 3), 1, dtype=wp.int32, device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                dofs_per_robot=dofs_2d,
                stiffness=_gains(6, 1.0, device),
                damping=_gains(6, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_2d_output_index_raises(self):
        """Verify a 2-D joint_f_idx raises even when its total element count matches."""
        device = wp.get_device()
        # Shape (2, 1) has size 2, matching sum(dofs_per_robot) for two 1-DOF robots.
        indices_2d = wp.array(np.arange(2, dtype=np.int32).reshape(2, 1), dtype=wp.int32, device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                dofs_per_robot=_dofs_arr([1, 1], device),
                joint_f_idx=indices_2d,
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_2d_index_override_raises(self):
        """Verify a 2-D per-port index override raises."""
        device = wp.get_device()
        indices_2d = wp.array(np.arange(2, dtype=np.int32).reshape(2, 1), dtype=wp.int32, device=device)
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                dofs_per_robot=_dofs_arr([1, 1], device),
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                joint_q_idx=indices_2d,
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_short_input_array_raises(self):
        """Verify an input array too short for its indices raises instead of reading out of bounds."""
        device = wp.get_device()
        # Indices reach slot 5, so any bound array must hold at least 6 entries.
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr([2], device),
            joint_q_idx=wp.array([0, 5], dtype=wp.int32, device=device),
            stiffness=_gains(2, 1.0, device),
            damping=_gains(2, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins, outs = ctrl.input(), ctrl.output()
        self.assertEqual(ins.joint_q.shape, (6,))
        ins.joint_q = wp.zeros(2, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_device_input_raises(self):
        """Verify an input array on another device raises instead of being dereferenced."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("needs a second device to mismatch against")
        ctrl = _make_mf(dofs_list=[2], kp=1.0, kd=0.0, device=device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q_des = wp.array([1.0, 1.0], dtype=wp.float32, device="cpu")
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_shape_live_gain_raises(self):
        """Verify a live gain whose shape differs from (controlled_dof_count,) raises."""
        device = wp.get_device()
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr([2, 2], device),
            stiffness=None,  # live: read from inputs.stiffness each step
            damping=_gains(4, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins, outs = ctrl.input(), ctrl.output()
        ins.stiffness = wp.full(1, 7.0, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_shape_mass_matrix_raises(self):
        """Verify a mass matrix whose shape differs from (robot_count, max_dofs, max_dofs) raises."""
        device = wp.get_device()
        ctrl = _make_mf(dofs_list=[2, 2], kp=1.0, kd=0.0, device=device, use_inertia=True)
        ins, outs = ctrl.input(), ctrl.output()
        ins.mass_matrix = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_device_constructor_array_raises(self):
        """Verify every wp.array constructor argument is rejected when it is on the wrong device."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("needs a second device to mismatch against")
        other = "cpu"

        def _kwargs():
            return {
                "dofs_per_robot": _dofs_arr([2], device),
                "stiffness": _gains(2, 1.0, device),
                "damping": _gains(2, 0.0, device),
                "use_gravity_compensation": False,
                "use_coriolis_compensation": False,
                "use_inertia_decoupling": False,
                "device": device,
            }

        # One entry per wp.array argument, each moved to the wrong device in turn.
        wrong = {
            "dofs_per_robot": _dofs_arr([2], other),
            "stiffness": _gains(2, 1.0, other),
            "damping": _gains(2, 0.0, other),
            "joint_q_idx": _iota(2, other),
            "joint_qd_idx": _iota(2, other),
            "gravity_force_idx": _iota(2, other),
            "coriolis_force_idx": _iota(2, other),
            "joint_f_idx": _iota(2, other),
        }
        for name, bad_array in wrong.items():
            with self.subTest(argument=name), self.assertRaises(ValueError):
                ControllerJointImpedanceModelFree(**{**_kwargs(), name: bad_array})

    def test_wrong_dtype_constructor_array_raises(self):
        """Verify a wp.array constructor argument with the wrong dtype raises TypeError."""
        device = wp.get_device()
        with self.assertRaises(TypeError):
            ControllerJointImpedanceModelFree(
                dofs_per_robot=_dofs_arr([2], device),
                joint_f_idx=wp.zeros(2, dtype=wp.uint32, device=device),  # want int32
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )

    def test_zero_dof_robot_raises(self):
        """Verify a robot declaring zero DOFs raises at construction."""
        device = wp.get_device()
        with self.assertRaises(ValueError):
            ControllerJointImpedanceModelFree(
                dofs_per_robot=_dofs_arr([2, 0], device),
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
                device=device,
            )


# ---------------------------------------------------------------------------
# ControllerJointImpedanceModelFree — heterogeneous
# ---------------------------------------------------------------------------


class TestControllerJointImpedanceModelFreeHeterogeneous(unittest.TestCase):
    def test_heterogeneous_pd_torques(self):
        """Verify PD torques are correct for each robot with different DOF counts."""
        device = wp.get_device()
        # Robot 0: 2 DOFs, Kp=5; Robot 1: 1 DOF, Kp=5
        # Errors: robot0=[1,0], robot1=[2]  →  tau: robot0=[5,0], robot1=[10]
        dofs_list = [2, 1]
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr(dofs_list, device),
            stiffness=_gains(3, 5.0, device),  # 2 + 1 = 3 controlled DOFs, no padding
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0, 0.0, 2.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau, [5.0, 0.0, 10.0], atol=1e-5)

    def test_heterogeneous_independence(self):
        """Verify robot 0's torques are zero when only robot 1 has a position error."""
        device = wp.get_device()
        dofs_list = [2, 1]
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr(dofs_list, device),
            stiffness=_gains(3, 1.0, device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([0.0, 0.0, 3.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        tau = outs.joint_f.numpy()
        # Only robot 1's slot (index 2) should be nonzero
        np.testing.assert_allclose(tau[:2], [0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(tau[2], 3.0, places=5)

    def test_heterogeneous_mass_matrix_padding_ignored(self):
        """Verify that padded mass-matrix entries never reach the torque output.

        Joint-space vectors are flat and ragged, so the mass matrix is the only
        padded buffer left. A short robot's unused rows and columns are filled
        with values that would visibly corrupt the result if they were read.
        """
        device = wp.get_device()
        # Robot 0 has 3 DOFs, robot 1 has 1 DOF, so M is padded to (2, 3, 3).
        dofs_list = [3, 1]
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr(dofs_list, device),
            stiffness=_gains(4, 1.0, device),
            damping=_gains(4, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
            device=device,
        )
        M_np = np.full((2, 3, 3), 1e6, dtype=np.float32)  # poison every padded entry
        M_np[0] = np.eye(3, dtype=np.float32)
        M_np[1, 0, 0] = 1.0
        ins = ctrl.input()
        ins.joint_q = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(4, dtype=wp.float32, device=device)
        # Distinct targets: the short robot's value must come from its own slot.
        ins.joint_q_des = wp.array([1.0, 2.0, 3.0, 99.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(4, dtype=wp.float32, device=device)
        ins.mass_matrix = wp.array(M_np, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        tau = outs.joint_f.numpy()
        self.assertEqual(tau.shape[0], 4)
        np.testing.assert_allclose(tau, [1.0, 2.0, 3.0, 99.0], atol=1e-3)

    def test_heterogeneous_inertia_decoupling(self):
        """Verify M @ acc is computed per-robot with heterogeneous DOF counts."""
        device = wp.get_device()
        # Robot 0: 2 DOFs, M=2*I; Robot 1: 1 DOF, M=[[3]]
        # Errors are distinct per DOF so that reading another robot's slice, or
        # dropping the per-robot offset into the flat vector, changes the result.
        # acc=[1,2] and [5]  ->  robot0 = 2*I @ [1,2] = [2,4], robot1 = [[3]] @ [5] = [15]
        dofs_list = [2, 1]
        ctrl = ControllerJointImpedanceModelFree(
            dofs_per_robot=_dofs_arr(dofs_list, device),
            stiffness=_gains(3, 1.0, device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
            device=device,
        )
        # Mass matrices padded to (2, 2, 2); robot1's second row/col is unused
        M_np = np.zeros((2, 2, 2), dtype=np.float32)
        M_np[0] = np.eye(2) * 2.0
        M_np[1, 0, 0] = 3.0
        M = wp.array(M_np, dtype=wp.float32, device=device)
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0, 2.0, 5.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        ins.mass_matrix = M
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau, [2.0, 4.0, 15.0], atol=1e-5)


# ---------------------------------------------------------------------------
# ControllerJointImpedance (model-based)
# ---------------------------------------------------------------------------


class TestControllerJointImpedance(unittest.TestCase):
    def _make_ctrl(self, device, *, kp=10.0, kd=1.0, use_inertia=False):
        """Build a ControllerJointImpedance for a single prismatic robot."""
        model = _build_single_prismatic()
        return ControllerJointImpedance(
            model,
            stiffness=_gains(1, kp, device),
            damping=_gains(1, kd, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=use_inertia,
        )

    def _run(self, ctrl, *, q_sim, qd_sim, q_des_sim, qd_des_sim, device):
        """Run one step and return the torque array."""
        ins = ctrl.input()
        ins.joint_q = wp.array(np.array(q_sim, dtype=np.float32), dtype=wp.float32, device=device)
        ins.joint_qd = wp.array(np.array(qd_sim, dtype=np.float32), dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array(np.array(q_des_sim, dtype=np.float32), dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.array(np.array(qd_des_sim, dtype=np.float32), dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        return outs.joint_f.numpy()

    def test_zero_error_gives_zero_torque(self):
        """Verify zero position and velocity error produces zero torque."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        tau = self._run(ctrl, q_sim=[0.5], qd_sim=[0.0], q_des_sim=[0.5], qd_des_sim=[0.0], device=device)
        np.testing.assert_allclose(tau, [0.0], atol=1e-4)

    def test_position_error_produces_stiffness_torque(self):
        """Verify τ = Kp * (q_des - q) for a simple prismatic robot."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device, kp=5.0, kd=0.0)
        tau = self._run(ctrl, q_sim=[0.0], qd_sim=[0.0], q_des_sim=[1.0], qd_des_sim=[0.0], device=device)
        np.testing.assert_allclose(tau, [5.0], atol=1e-4)

    def test_damping_term(self):
        """Verify τ = Kd * (qd_des - qd) when Kp=0."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device, kp=0.0, kd=3.0)
        tau = self._run(ctrl, q_sim=[0.0], qd_sim=[0.0], q_des_sim=[0.0], qd_des_sim=[2.0], device=device)
        np.testing.assert_allclose(tau, [6.0], atol=1e-4)

    def test_is_graphable_true(self):
        """Verify is_graphable() returns True."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        self.assertTrue(ctrl.is_graphable())

    def test_ball_joint_raises(self):
        """Verify that a multi-DOF ball joint raises ValueError."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        j = builder.add_joint_ball(
            parent=-1,
            child=link,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j], label="ball_robot")
        model = builder.finalize()
        with self.assertRaises(ValueError):
            ControllerJointImpedance(
                model,
                stiffness=_gains(3, 1.0, device),
                damping=_gains(3, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
            )

    def test_fixed_joint_allowed(self):
        """Verify that fixed joints (zero DOF) are accepted alongside revolute/prismatic joints."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        base = builder.add_link()
        arm = builder.add_link()
        j_fixed = builder.add_joint_fixed(
            parent=-1,
            child=base,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        j_rev = builder.add_joint_revolute(
            parent=base,
            child=arm,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j_fixed, j_rev], label="robot")
        model = builder.finalize()
        # Should not raise — fixed joint is zero-DOF and irrelevant to the PD term.
        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(1, 10.0, device),
            damping=_gains(1, 1.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
        )
        self.assertIsNotNone(ctrl)

    def test_model_is_borrowed_not_copied(self):
        """Verify the controller holds the caller's model rather than building its own.

        Construction takes a finalized model and must not duplicate or replace it,
        so runtime changes to the model are visible to the controller.
        """
        device = wp.get_device()
        model = _build_single_prismatic()
        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(1, 10.0, device),
            damping=_gains(1, 1.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertIs(ctrl._model, model)
        self.assertEqual(ctrl.device, model.device)

    def test_uncontrolled_joint_in_articulation_affects_gravity(self):
        """Verify an uncontrolled joint inside a controlled articulation drives FK.

        A 1-DOF arm is mounted on a "door" that tilts about a horizontal axis. The
        door belongs to the articulation but is excluded from the selection, so it
        receives no torque — yet its measured position changes the arm's
        orientation relative to gravity, so gravity compensation must follow it.
        """
        device = wp.get_device()
        model = _build_arm_on_door(door_in_articulation=True)

        ctrl = ControllerJointImpedance(
            model,
            joints=["arm"],  # the door joint is read for FK, never actuated
            stiffness=_gains(1, 0.0, device),  # gravity compensation only
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=True,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.controlled_joints, (1,))

        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_qd = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.zeros(1, dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(1, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(2, dtype=wp.float32, device=device)

        torques = []
        for door_q in (0.0, 1.0):
            ins.joint_q = wp.array([door_q, 0.6], dtype=wp.float32, device=device)
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)
            torques.append(float(outs.joint_f.numpy()[1]))

        # Tilting the door through 1 rad flips the sign of the arm's gravity torque.
        self.assertLess(torques[0], -0.4)
        self.assertGreater(torques[1], -0.1)
        # The door's own slot is never written.
        self.assertEqual(float(outs.joint_f.numpy()[0]), 0.0)

    def test_inertia_uses_the_selected_articulation_block(self):
        """Verify the mass matrix is taken from the selected articulation, at the right offset.

        Articulation 1 begins at model DOF 2 and has a different geometry from
        articulation 0, so using the wrong block or dropping the per-articulation
        DOF base both produce a visibly different torque.
        """
        device = wp.get_device()
        model = _build_two_arms_with_mass()
        self.assertEqual(model.articulation_count, 2)

        ctrl = ControllerJointImpedance(
            model,
            articulations=[1],
            stiffness=_gains(2, 1.0, device),
            damping=_gains(2, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
        )
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(4, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0, 0.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(2, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(4, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        # Expected: H[articulation 1] @ [1, 0], i.e. that block's first column.
        state = model.state()
        state.joint_q.zero_()
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        H = newton.eval_mass_matrix(model, state).numpy()
        expected = H[1][:2, 0]

        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau[2:], expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(tau[:2], [0.0, 0.0], atol=1e-6)
        # The two articulations must genuinely differ, or this proves nothing.
        self.assertFalse(np.allclose(H[0][:2, 0], expected, rtol=1e-3))

    def test_model_with_stray_fixed_joint_accepted(self):
        """Verify a zero-DOF joint outside any articulation does not trip the DOF check.

        Ground planes and rigid welds contribute no DOFs, so the check must not
        reject them.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder()
        arm = builder.add_link()
        j = builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j], label="robot")
        weld = builder.add_link()
        builder.add_joint_fixed(
            parent=-1,
            child=weld,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_ground_plane()
        model = builder.finalize()

        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(1, 10.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertIsNotNone(ctrl)

    def test_articulation_selection_controls_subset(self):
        """Verify `articulations` restricts control to the selected articulations."""
        device = wp.get_device()
        model = _build_two_robot_mixed()  # articulation 0: 2 DOFs, articulation 1: 1 DOF

        ctrl = ControllerJointImpedance(
            model,
            articulations=[1],  # the 1-DOF robot only
            stiffness=_gains(1, 4.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.robot_count, 1)
        self.assertEqual(ctrl.controlled_dof_count, 1)

        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([2.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(1, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(3, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau[2], 8.0, atol=1e-4)  # 4.0 * 2.0
        np.testing.assert_allclose(tau[:2], [0.0, 0.0], atol=1e-6)  # unselected robot untouched

    def test_articulation_selection_rejects_bad_indices(self):
        """Verify out-of-range, duplicate, and empty `articulations` are rejected."""
        device = wp.get_device()
        model = _build_two_robot_mixed()

        def _make(selection):
            return ControllerJointImpedance(
                model,
                articulations=selection,
                stiffness=_gains(1, 1.0, device),
                damping=_gains(1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

        for bad in ([5], [0, 0], [], [-1]):
            with self.subTest(articulations=bad), self.assertRaises(ValueError):
                _make(bad)

    def test_joint_selection_controls_subset_of_a_robot(self):
        """Verify `joints` restricts control to some joints of one articulation.

        Robot 0 has two revolute DOFs; only its second is controlled. The first
        must be left alone even though it belongs to the same articulation.
        """
        device = wp.get_device()
        model = _build_two_link_arm_with_mass()  # one articulation, 2 DOFs, real inertia
        ctrl = ControllerJointImpedance(
            model,
            joints=[1],  # only the second joint of that articulation
            stiffness=_gains(1, 2.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,  # exercises the mass-matrix submatrix path
        )
        self.assertEqual(ctrl.robot_count, 1)
        self.assertEqual(ctrl.max_dofs, 1)
        self.assertEqual(ctrl.controlled_joints, (1,))

        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(2, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(1, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(2, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        tau = outs.joint_f.numpy()
        self.assertNotEqual(float(tau[1]), 0.0)  # the controlled DOF got a torque
        np.testing.assert_allclose(tau[0], 0.0, atol=1e-6)  # its sibling DOF is untouched

    def test_joint_selection_rejects_bad_entries(self):
        """Verify empty, duplicate, and out-of-scope `joints` entries are rejected."""
        device = wp.get_device()
        model = _build_two_robot_mixed()

        def _make(selection):
            return ControllerJointImpedance(
                model,
                articulations=[0],
                joints=selection,
                stiffness=_gains(1, 1.0, device),
                damping=_gains(1, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

        # empty, duplicated, outside the selected articulation, nonexistent label
        for bad in ([], [0, 0], [2], ["nope"]):
            with self.subTest(joints=bad), self.assertRaises(ValueError):
                _make(bad)

    def test_joint_selection_by_label(self):
        """Verify a joint label selects the matching joint of every replicated robot."""
        device = wp.get_device()
        arm = newton.ModelBuilder()
        prev = -1
        joints = []
        for label in ("shoulder", "elbow"):
            link = arm.add_link()
            joints.append(arm.add_joint_revolute(parent=prev, child=link, axis=wp.vec3(0.0, 0.0, 1.0), label=label))
            arm.add_shape_box(body=link, hx=0.02, hy=0.1, hz=0.02)
            prev = link
        arm.add_articulation(joints, label="arm")
        scene = newton.ModelBuilder()
        scene.replicate(arm, world_count=3, spacing=(1.0, 0.0, 0.0))
        model = scene.finalize()

        ctrl = ControllerJointImpedance(
            model,
            joints=["elbow"],  # one string addresses all three robots
            stiffness=_gains(3, 1.0, device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.robot_count, 3)
        self.assertEqual(ctrl.controlled_joints, (1, 3, 5))

    def test_uncontrolled_non_scalar_joint_allowed(self):
        """Verify a free joint elsewhere in the model does not block construction.

        Only the controlled joints must be scalar. A floating base is read for
        forward kinematics and never actuated, so it is legal. Its presence also
        makes coordinate and DOF indices diverge, which the controller must handle.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder()
        torso = builder.add_link()
        j_free = builder.add_joint_free(child=torso)
        builder.add_shape_box(body=torso, hx=0.2, hy=0.1, hz=0.1)
        link = builder.add_link()
        j_rev = builder.add_joint_revolute(parent=torso, child=link, axis=wp.vec3(0.0, 0.0, 1.0), label="shoulder")
        builder.add_shape_box(body=link, hx=0.02, hy=0.1, hz=0.02)
        builder.add_articulation([j_free, j_rev], label="mobile")
        model = builder.finalize()
        self.assertNotEqual(model.joint_coord_count, model.joint_dof_count)

        ctrl = ControllerJointImpedance(
            model,
            joints=["shoulder"],
            stiffness=_gains(1, 3.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.controlled_dof_count, 1)

        # The arm sits at coordinate 7 but DOF 6; reading the wrong one picks up
        # a quaternion component of the floating base instead of the joint angle.
        ins, outs = ctrl.input(), ctrl.output()
        q = np.zeros(model.joint_coord_count, dtype=np.float32)
        q[7] = 0.5  # the shoulder angle
        ins.joint_q = wp.array(q, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.zeros(1, dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(1, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        # tau = Kp * (0 - 0.5) = -1.5, written at the arm's DOF slot 6.
        np.testing.assert_allclose(outs.joint_f.numpy()[6], -1.5, atol=1e-4)

    def test_externally_mounted_articulation_raises(self):
        """Verify an articulation mounted on a body it does not drive is rejected.

        Newton's inverse dynamics never writes scratch state for such a body, so
        gravity and Coriolis terms would read uninitialised memory and differ
        between runs. A zero-DOF mount is corrupted just as badly as a revolute
        one, so both must be rejected.
        """
        device = wp.get_device()
        for mount_fixed in (True, False):
            model = _build_arm_on_external_mount(mount_fixed=mount_fixed)
            with self.subTest(mount_fixed=mount_fixed), self.assertRaisesRegex(ValueError, "rooted at the world"):
                ControllerJointImpedance(
                    model,
                    stiffness=_gains(1, 1.0, device),
                    damping=_gains(1, 0.0, device),
                    use_gravity_compensation=True,
                    use_coriolis_compensation=False,
                    use_inertia_decoupling=False,
                )

    def test_externally_mounted_articulation_allowed_without_dynamics(self):
        """Verify the rooting check applies only when dynamics are actually evaluated.

        With every compensation disabled the controller is a pure PD law: it never
        calls inverse dynamics, so the uninitialised-scratch hazard cannot arise
        and the model must not be rejected.
        """
        device = wp.get_device()
        model = _build_arm_on_external_mount(mount_fixed=True)
        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(1, 2.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(1, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        # The fixed mount contributes no DOF, so the arm owns the model's only one.
        arm_dof = int(model.joint_qd_start.numpy()[1])
        np.testing.assert_allclose(outs.joint_f.numpy()[arm_dof], 2.0, atol=1e-5)

    def test_articulation_mounted_on_another_articulation_raises(self):
        """Verify a cross-articulation mount is rejected even though both are driven.

        Each articulation is traversed independently, so a body driven by a
        *different* articulation is still unwritten when this one is evaluated.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder()
        torso = builder.add_link()
        j_torso = builder.add_joint_revolute(
            parent=-1,
            child=torso,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_box(body=torso, hx=0.25, hy=0.02, hz=0.02)
        builder.add_articulation([j_torso], label="torso")
        arm = builder.add_link()
        j_arm = builder.add_joint_revolute(
            parent=torso,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_box(body=arm, hx=0.15, hy=0.02, hz=0.02)
        builder.add_articulation([j_arm], label="arm")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "rooted at the world"):
            ControllerJointImpedance(
                model,
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=True,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

    def test_non_root_joint_with_external_parent_raises(self):
        """Verify the rooting check covers every joint, not just the articulation's root.

        An articulation may be a forest: its first joint can be world-rooted while a
        later one hangs off a body driven elsewhere. Checking only the root would
        pass this model and then read uninitialised state for that body.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder()
        external = builder.add_link()
        builder.add_joint_revolute(parent=-1, child=external, axis=wp.vec3(0.0, 1.0, 0.0))
        builder.add_shape_box(body=external, hx=0.1, hy=0.02, hz=0.02)
        first = builder.add_link()
        j_first = builder.add_joint_revolute(parent=-1, child=first, axis=wp.vec3(0.0, 1.0, 0.0))
        builder.add_shape_box(body=first, hx=0.1, hy=0.02, hz=0.02)
        second = builder.add_link()
        j_second = builder.add_joint_revolute(parent=external, child=second, axis=wp.vec3(0.0, 1.0, 0.0))
        builder.add_shape_box(body=second, hx=0.1, hy=0.02, hz=0.02)
        builder.add_articulation([j_first, j_second], label="forest")
        model = builder.finalize()

        with self.assertRaisesRegex(ValueError, "rooted at the world"):
            ControllerJointImpedance(
                model,
                articulations=["forest"],
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=True,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

    def test_over_matching_label_is_caught_by_the_gain_length(self):
        """Verify selecting more joints than intended fails on the gain array's shape.

        Labels match every joint carrying them, so a label shared by two robot
        kinds selects both. That is deliberate, and a caller who expected only one
        kind is caught here rather than by a heuristic: the flat gain array must
        be exactly ``controlled_dof_count`` long, and the error names the real
        count.
        """
        device = wp.get_device()
        # Two robot kinds, both with a default-labelled `joint_1`: four matches.
        scene = newton.ModelBuilder()
        for links in (3, 2):
            arm = newton.ModelBuilder()
            prev, joints = -1, []
            for _ in range(links):
                link = arm.add_link()
                joints.append(arm.add_joint_revolute(parent=prev, child=link, axis=wp.vec3(0.0, 0.0, 1.0)))
                arm.add_shape_box(body=link, hx=0.02, hy=0.1, hz=0.02)
                prev = link
            arm.add_articulation(joints)
            scene.replicate(arm, world_count=2, spacing=(1.0, 0.0, 0.0))
        model = scene.finalize()

        with self.assertRaisesRegex(ValueError, r"stiffness must have shape \(4\), got \(2,\)"):
            ControllerJointImpedance(
                model,
                joints=["joint_1"],  # the caller believes this is one robot kind
                stiffness=_gains(2, 1.0, device),
                damping=_gains(2, 0.0, device),
                use_gravity_compensation=False,
                use_coriolis_compensation=False,
                use_inertia_decoupling=False,
            )

    def test_output_is_sized_for_the_scattered_slots(self):
        """Verify output() allocates for the simulation slots, not the controlled count.

        Controlling one DOF that lands at simulation slot 2 needs a length-3
        buffer; sizing it by the controlled DOF count would return a length-1
        array that the scatter would immediately overrun.
        """
        device = wp.get_device()
        model = _build_two_robot_mixed()  # articulation 1 owns model DOF 2
        ctrl = ControllerJointImpedance(
            model,
            articulations=[1],
            stiffness=_gains(1, 1.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        self.assertEqual(ctrl.controlled_dof_count, 1)
        self.assertEqual(ctrl.output().joint_f.shape, (3,))

    def test_wrong_mapping_shape_raises(self):
        """Verify each model-to-simulation mapping must have one entry per model coordinate or DOF."""
        device = wp.get_device()
        model = _build_two_robot_mixed()  # 3 coordinates, 3 DOFs
        for bad in ("model_coord_to_sim_coord", "model_dof_to_sim_dof"):
            with self.subTest(argument=bad), self.assertRaises(ValueError):
                ControllerJointImpedance(
                    model,
                    stiffness=_gains(3, 1.0, device),
                    damping=_gains(3, 0.0, device),
                    use_gravity_compensation=False,
                    use_coriolis_compensation=False,
                    use_inertia_decoupling=False,
                    **{bad: _iota(5, device)},  # want 3
                )

    def test_mapping_remaps_sim_layout(self):
        """Verify a non-identity mapping redirects both reads and writes.

        With the simulation array laid out in reverse relative to the model, the
        controller must read each model DOF from its remapped slot and scatter the
        torque back to the same place.
        """
        device = wp.get_device()
        model = _build_two_robot_mixed()  # 3 DOFs: robot0 at 0,1 and robot1 at 2
        reversed_layout = wp.array([2, 1, 0], dtype=wp.int32, device=device)
        ctrl = ControllerJointImpedance(
            model,
            articulations=[1],  # the 1-DOF robot, model DOF 2 -> sim slot 0
            model_coord_to_sim_coord=reversed_layout,
            model_dof_to_sim_dof=reversed_layout,
            stiffness=_gains(1, 3.0, device),
            damping=_gains(1, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([5.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(1, dtype=wp.float32, device=device)
        outs.joint_f = wp.zeros(3, dtype=wp.float32, device=device)  # a real sim-sized buffer
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        tau = outs.joint_f.numpy()
        np.testing.assert_allclose(tau[0], 15.0, atol=1e-4)  # 3.0 * 5.0, at the remapped slot
        np.testing.assert_allclose(tau[1:], [0.0, 0.0], atol=1e-6)

    def test_short_joint_q_raises_before_gather(self):
        """Verify a short joint_q is rejected before the internal FK gather reads it."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(0, dtype=wp.float32, device=device)
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_wrong_device_joint_q_raises_before_gather(self):
        """Verify a joint_q on another device is rejected before the internal FK gather reads it."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("needs a second device to mismatch against")
        ctrl = self._make_ctrl(device)
        ins, outs = ctrl.input(), ctrl.output()
        ins.joint_q = wp.zeros(1, dtype=wp.float32, device="cpu")
        with self.assertRaises(ValueError):
            ctrl.step(inputs=ins, outputs=outs, dt=0.01)

    def test_input_outputs_shapes(self):
        """Verify input/output struct arrays have the expected flat shapes."""
        device = wp.get_device()
        ctrl = self._make_ctrl(device)
        ins = ctrl.input()
        outs = ctrl.output()
        self.assertEqual(ins.joint_q.shape, (1,))
        self.assertEqual(outs.joint_f.shape, (1,))

    def test_heterogeneous_model(self):
        """Verify model-based controller works with a heterogeneous two-robot fleet."""
        device = wp.get_device()
        model = _build_two_robot_mixed()  # robot0: 2 DOFs, robot1: 1 DOF
        ctrl = ControllerJointImpedance(
            model,
            stiffness=_gains(3, 4.0, device),  # 2 + 1 controlled DOFs, no padding
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
        )
        ins = ctrl.input()
        ins.joint_q = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_qd = wp.zeros(3, dtype=wp.float32, device=device)
        ins.joint_q_des = wp.array([1.0, 0.0, 2.0], dtype=wp.float32, device=device)
        ins.joint_qd_des = wp.zeros(3, dtype=wp.float32, device=device)
        outs = ctrl.output()
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)
        tau = outs.joint_f.numpy()
        # robot0 DOF0: 4*1=4, robot0 DOF1: 4*0=0, robot1 DOF0: 4*2=8
        np.testing.assert_allclose(tau, [4.0, 0.0, 8.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
