# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Heterogeneous Joint Impedance
#
# Four articulations, showing two ways to leave a joint uncontrolled:
#   0  arm_3dof      — 3 DOFs, ALL CONTROLLED, holds itself out against gravity
#   1  pendulum      — 1 DOF,  NONE CONTROLLED, swings freely and damps out
#   2  arm_1dof      — 1 DOF,  ALL CONTROLLED, rotates continuously
#   3  arm_gripper   — 3 DOFs, 2 CONTROLLED: the gripper joint is left free
#
# The uncontrolled pendulum sits *between* the two controlled arms, so whole
# articulations are skipped from the middle of the index range. The gripper goes
# further: one joint *inside* an otherwise-controlled articulation is excluded.
# Both are still read every step, so forward kinematics stays correct, but no
# torque is ever written to their slots.
#
# Every joint rotates about Y with its links extending along +X, so the arms
# swing in a vertical plane and carry their own weight. The gains are soft
# enough that the 3-DOF arm visibly sags if gravity compensation is turned off —
# it is the model-based term, not stiffness, that holds the arm out.
#
# The two controlled arms have different DOF counts, exercising the ragged,
# unpadded heterogeneous layout.
#
# Command: python -m newton.examples controller_joint_impedance_heterogeneous
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton import JointTargetMode
from newton.controllers import ControllerJointImpedance, select_joints

# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------

LINK_LEN_A = 0.25  # length of each link in the 3-DOF arm [m]
LINK_LEN_PENDULUM = 0.45  # length of the uncontrolled pendulum's link [m]
LINK_LEN_B = 0.45  # length of the 1-DOF arm's link [m]
LINK_LEN_G = 0.25  # length of each link in the gripper arm [m]
LINK_LEN_FINGER = 0.12  # length of the gripper's free finger [m]
BASE_HEIGHT = 0.8  # height of every base above the ground plane [m]

PENDULUM_DAMPING = 0.35  # viscous damping on the free pendulum [N·m·s/rad]

DOFS_A = 3
DOFS_PENDULUM = 1
DOFS_B = 1
DOFS_G = 2  # controlled joints of the gripper arm, excluding the finger
CONTROLLED_DOFS = DOFS_A + DOFS_B + DOFS_G  # = 6
TOTAL_DOFS = DOFS_A + DOFS_PENDULUM + DOFS_B + DOFS_G + 1  # = 8 model DOFs

# Model DOF slots, in build order.
DOF_SLOTS_A = [0, 1, 2]
DOF_SLOT_PENDULUM = 3
DOF_SLOT_B = 4
DOF_SLOTS_G = [5, 6]
DOF_SLOT_FINGER = 7

# Gains — one entry per *controlled* DOF, robot-major: [A0, A1, A2, B0].
# Heterogeneous fleets need no padding, and the uncontrolled pendulum has no
# entry at all. Units are 1/s² and 1/s because inertia decoupling is enabled.
#
# Deliberately soft, so the gravity compensation carries the arm rather than
# brute stiffness. Measured at t=5 s: joint error peaks at 0.01 rad with
# compensation on and 0.46 rad with it off, dropping the arm's tip 12 cm.
KP = np.full(CONTROLLED_DOFS, 60.0, dtype=np.float32)
KD = np.full(CONTROLLED_DOFS, 14.0, dtype=np.float32)

# The 3-DOF arm holds itself out horizontally, with a gentle wave on each joint.
TARGET_AMP = 0.25  # [rad]
TARGET_FREQ = 0.25  # [Hz]
PHASE_A = [0.0, math.pi / 3, 2 * math.pi / 3]

ROT_SPEED_B = 1.5  # continuous rotation rate of the 1-DOF arm [rad/s]


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _add_arm(builder, n_links, link_len, x_offset, label, finger_len=None):
    """Add an n-link arm whose joints rotate about Y, so it swings in the XZ plane.

    Links extend along +X from their joint, so a zero joint angle holds the arm
    straight out horizontally — the pose that costs the most gravity torque.

    Args:
        finger_len: Length of an extra short link appended as a "gripper" joint,
            or ``None`` for a plain arm. Returned last, so ``joints[:-1]`` are
            the arm joints and ``joints[-1]`` is the gripper.
    """
    joints = []
    prev_body = -1
    for i in range(n_links):
        body = builder.add_link()

        # Pivot: the elevated base for the first link, the end of the previous one after that.
        if i == 0:
            parent_xform = wp.transform(wp.vec3(x_offset, 0.0, BASE_HEIGHT), wp.quat_identity())
        else:
            parent_xform = wp.transform(wp.vec3(link_len, 0.0, 0.0), wp.quat_identity())

        joints.append(
            builder.add_joint_revolute(
                parent=prev_body,
                child=body,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
                label=f"{label}_j{i}",
            )
        )

        # Thin rod centred at half the link length out along +X.
        builder.add_shape_box(
            body=body,
            xform=wp.transform(wp.vec3(link_len * 0.5, 0.0, 0.0), wp.quat_identity()),
            hx=link_len * 0.5,
            hy=0.02,
            hz=0.02,
        )

        prev_body = body

    if finger_len is not None:
        finger = builder.add_link()
        joints.append(
            builder.add_joint_revolute(
                parent=prev_body,
                child=finger,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(link_len, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                label=f"{label}_gripper",
            )
        )
        builder.add_shape_box(
            body=finger,
            xform=wp.transform(wp.vec3(finger_len * 0.5, 0.0, 0.0), wp.quat_identity()),
            hx=finger_len * 0.5,
            hy=0.02,
            hz=0.02,
        )

    builder.add_articulation(joints, label=label)
    return joints


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    @staticmethod
    def create_parser():
        return newton.examples.create_parser()

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Physics scene ---------------------------------------------------
        # Build order fixes the articulation indices: the uncontrolled pendulum
        # is articulation 1, between the two controlled arms.
        scene = newton.ModelBuilder()
        _add_arm(scene, DOFS_A, LINK_LEN_A, x_offset=-1.0, label="arm_3dof")
        _add_arm(scene, DOFS_PENDULUM, LINK_LEN_PENDULUM, x_offset=0.0, label="pendulum")
        _add_arm(scene, DOFS_B, LINK_LEN_B, x_offset=1.0, label="arm_1dof")
        _add_arm(scene, DOFS_G, LINK_LEN_G, x_offset=2.0, label="arm_gripper", finger_len=LINK_LEN_FINGER)
        scene.add_ground_plane()

        for i in range(TOTAL_DOFS):
            scene.joint_target_ke[i] = 0.0
            scene.joint_target_kd[i] = 0.0
            scene.joint_target_mode[i] = int(JointTargetMode.EFFORT)

        # The free pendulum would swing forever otherwise. Damping is a property of
        # the model, not of the controller, so it applies to uncontrolled joints too.
        scene.joint_damping[DOF_SLOT_PENDULUM] = PENDULUM_DAMPING

        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # ---- Impedance controller --------------------------------------------
        # Two different exclusions in one selection: the pendulum articulation is
        # never named, and the gripper joint is dropped from the arm that owns it.
        # `exclude` takes labels, so this stays the same length whether there is
        # one gripper arm or a thousand.
        controlled = select_joints(
            self.model,
            articulations=["arm_3dof", "arm_1dof", "arm_gripper"],
            exclude_joints=["arm_gripper_gripper"],
        )
        self.controller = ControllerJointImpedance(
            self.model,
            joints=controlled,
            stiffness=wp.array(KP, dtype=wp.float32, device=self.device),
            damping=wp.array(KD, dtype=wp.float32, device=self.device),
            use_gravity_compensation=True,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
        )

        self._input = self.controller.input()
        self._output = self.controller.output()
        # Wire torque output directly into the sim control buffer.
        self._output.joint_f = self.control.joint_f

        # Bind live sim arrays before capture so the graph records the correct
        # buffer addresses. state_0 holds the current frame result after
        # sim_substeps (even number), so these pointers remain valid each replay.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd

        self._graph = None
        if self.controller.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._gpu_step()
            self._graph = capture.graph

        self.viewer.set_model(self.model)

    def _gpu_step(self):
        """Pure GPU work: controller step + physics substeps. Safe to graph-capture."""
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.sim_dt)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _targets(self):
        """Return desired positions and velocities, one entry per controlled DOF.

        The velocity target matters: leaving it at zero while the position target
        moves makes the damping term fight the motion, and the tracking lag settles
        at ``Kd * v / Kp`` — 0.35 rad for the rotating arm at these gains.
        """
        q_des = np.zeros(CONTROLLED_DOFS, dtype=np.float32)
        qd_des = np.zeros(CONTROLLED_DOFS, dtype=np.float32)
        omega = 2.0 * math.pi * TARGET_FREQ
        for k, phase in enumerate(PHASE_A):
            q_des[k] = TARGET_AMP * math.sin(omega * self.sim_time + phase)
            qd_des[k] = TARGET_AMP * omega * math.cos(omega * self.sim_time + phase)
        q_des[DOFS_A] = ROT_SPEED_B * self.sim_time  # the 1-DOF arm rotates continuously
        qd_des[DOFS_A] = ROT_SPEED_B
        # The gripper arm holds its two controlled joints straight out; its finger
        # has no target because it has no gain and no torque.
        return q_des, qd_des

    def step(self):
        # Update targets on the CPU — cannot be graph-captured.
        q_des, qd_des = self._targets()
        self._input.joint_q_des.assign(q_des)
        self._input.joint_qd_des.assign(qd_des)

        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._gpu_step()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the controlled joints track while the excluded ones stay unactuated."""
        joint_q = self.state_0.joint_q.numpy()
        joint_f = self.control.joint_f.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"

        # Neither excluded joint may receive a torque: a whole uncontrolled
        # articulation, and one joint inside a controlled one.
        for name, slot in (("pendulum", DOF_SLOT_PENDULUM), ("gripper finger", DOF_SLOT_FINGER)):
            assert joint_f[slot] == 0.0, f"uncontrolled {name} was actuated: {joint_f[slot]}"

        # ...and both must actually have moved, or this proves nothing about selection.
        pendulum_q = float(joint_q[DOF_SLOT_PENDULUM])
        assert abs(pendulum_q) > 0.2, f"pendulum did not swing: q={pendulum_q:.3f}"
        finger_q = float(joint_q[DOF_SLOT_FINGER])
        assert abs(finger_q) > 0.05, f"gripper finger did not hang free: q={finger_q:.3f}"

        # Damping must have bled the pendulum's swing away. It settles hanging
        # straight down (q = pi/2 here, since a zero angle holds the link out
        # horizontally), so "motion died out" is a statement about velocity.
        pendulum_qd = float(self.state_0.joint_qd.numpy()[DOF_SLOT_PENDULUM])
        assert abs(pendulum_qd) < 0.05, f"pendulum is still swinging: qd={pendulum_qd:.3f}"
        np.testing.assert_allclose(
            pendulum_q,
            math.pi / 2,
            atol=0.05,
            err_msg=f"pendulum did not settle hanging: q={pendulum_q:.3f}",
        )

        # The 3-DOF arm holds itself up: without gravity compensation its joints
        # would sag far past the commanded amplitude instead of tracking it.
        q_des, _ = self._targets()
        arm_a = joint_q[DOF_SLOTS_A]
        np.testing.assert_allclose(
            arm_a,
            q_des[:DOFS_A],
            atol=0.15,
            err_msg=f"3-DOF arm is not holding its target: q={arm_a}, target={q_des[:DOFS_A]}",
        )

        # The 1-DOF arm keeps up with its rotating target.
        np.testing.assert_allclose(
            joint_q[DOF_SLOT_B],
            q_des[DOFS_A],
            atol=0.15,
            err_msg=f"1-DOF arm not tracking: q={joint_q[DOF_SLOT_B]:.3f}, expected={q_des[DOFS_A]:.3f}",
        )

        # The gripper arm holds its two controlled joints out, carrying the weight
        # of a finger that contributes no torque of its own.
        arm_g = joint_q[DOF_SLOTS_G]
        np.testing.assert_allclose(
            arm_g,
            q_des[DOFS_A + DOFS_B :],
            atol=0.15,
            err_msg=f"gripper arm is not holding its target: q={arm_g}, target={q_des[DOFS_A + DOFS_B :]}",
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
