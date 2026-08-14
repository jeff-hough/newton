# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK over a heterogeneous fleet
#
# One ControllerDifferentialKinematics drives four robots of two different
# kinds toward draggable target frames:
#
#   0, 1  Franka FR3 + hand — 9-DOF articulations, 7 controlled
#   2, 3  Universal Robots UR10 — 6 DOF, all controlled
#
# The fleet is ragged in two independent ways. The two robot kinds have
# different DOF counts, and the Frankas are only partly controlled: their
# gripper fingers are excluded with select_joints(), so the fingers hold their
# home position while the arm tracks. The controller pads only the Jacobian;
# joint-space arrays stay flat, so nothing has to be padded by hand.
#
# The MuJoCo solver applies the commanded joint-position targets through its
# built-in PD. Because the controller writes only the controlled DOFs, the
# finger targets seeded at startup are never disturbed.
#
# Command: python -m newton.examples controller_diff_ik_heterogeneous
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.utils
from newton.controllers import CommandType, ControllerDifferentialKinematics, IkMethod, select_joints

# ---------------------------------------------------------------------------
# Fleet layout
# ---------------------------------------------------------------------------

FRANKA_COUNT = 2
UR10_COUNT = 2
ROBOT_COUNT = FRANKA_COUNT + UR10_COUNT
ROBOT_SPACING_Y = 1.4  # lateral gap between bases [m]

# Home poses. The Franka's last two entries are the gripper fingers, which the
# controller never drives.
FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=np.float32)
UR10_HOME = np.array([0.0, -1.2, 1.6, -1.9, -1.571, 0.0], dtype=np.float32)

FRANKA_FINGER_JOINTS = ["fr3/fr3_finger_joint1", "fr3/fr3_finger_joint2"]

# fr3_link7 -> fr3_link8 (z=+0.107) -> fr3_hand -> fr3_hand_tcp (z=+0.1034).
# Those fixed joints are collapsed at load time, so the TCP expressed in
# fr3_link7's frame is their sum.
FRANKA_TCP = wp.vec3(0.0, 0.0, 0.2104)
UR10_TCP = wp.vec3(0.0, 0.0, 0.1)  # tool flange ahead of wrist_3

# Joint-target PD gains, per robot kind. The UR10 is the heavier arm with the
# longer moment arm, so the stiffness that holds a Franka steady leaves it
# drooping ~7 cm — the IK commands the right target either way, but the PD has
# to be able to hold it.
FRANKA_GAINS = (3000.0, 100.0)
UR10_GAINS = (20000.0, 1500.0)
BANDWIDTH = 20.0


def _add_franka(builder):
    """Load one Franka FR3 with a hand, and tag its TCP with an ``ee`` site."""
    urdf = newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"
    builder.add_urdf(str(urdf), floating=False, collapse_fixed_joints=True)
    builder.add_site(
        builder.body_label.index("fr3/fr3_link7"),
        label="ee",
        xform=wp.transform(p=FRANKA_TCP, q=wp.quat_identity()),
        visible=True,
        scale=(0.02, 0.02, 0.02),
    )
    builder.joint_q = FRANKA_HOME.tolist()
    return builder


def _add_ur10(builder):
    """Load one UR10 and tag its tool flange with an ``ee`` site.

    Its root is a zero-DOF D6 joint, which the controller skips automatically:
    only 1-DOF revolute and prismatic joints are controllable.
    """
    usd = newton.utils.download_asset("universal_robots_ur10") / "usd" / "ur10_instanceable.usda"
    builder.add_usd(str(usd), collapse_fixed_joints=True, enable_self_collisions=False, hide_collision_shapes=True)
    builder.add_site(
        builder.body_label.index("/ur10/wrist_3_link"),
        label="ee",
        xform=wp.transform(p=UR10_TCP, q=wp.quat_identity()),
        visible=True,
        scale=(0.02, 0.02, 0.02),
    )
    builder.joint_q = UR10_HOME.tolist()
    return builder


def _template(loader, gains):
    """Build a one-robot template with its joint-target PD gains applied."""
    builder = newton.ModelBuilder()
    loader(builder)
    ke, kd = gains
    for i in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[i] = ke
        builder.joint_target_kd[i] = kd
    return builder


class Example:
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--transpose-solver",
            action="store_true",
            help="Use the Jacobian-transpose solver instead of damped least squares.",
        )
        return parser

    def __init__(self, viewer, args):
        # joint_target_q / joint_target_qd are the coordinate-layout arrays
        # MuJoCo's joint-target PD reads from.
        newton.use_coord_layout_targets = True

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Scene: two robot kinds in one world, in a row along Y ----------
        scene = newton.ModelBuilder()
        y = -0.5 * (ROBOT_COUNT - 1) * ROBOT_SPACING_Y
        fleet = [(_add_franka, FRANKA_GAINS)] * FRANKA_COUNT + [(_add_ur10, UR10_GAINS)] * UR10_COUNT
        for loader, gains in fleet:
            scene.add_builder(_template(loader, gains), xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()))
            y += ROBOT_SPACING_Y
        scene.add_ground_plane()

        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # The arms should not collide with anything here, and disabling contacts
        # keeps the demo focused on tracking.
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # ---- Controller ------------------------------------------------------
        # One label covers both Frankas, so excluding the grippers costs two
        # strings however many robots carry them. The UR10s have no joint with
        # these labels and are untouched by the exclusion.
        controlled = select_joints(self.model, exclude_joints=FRANKA_FINGER_JOINTS)
        self.controller = ControllerDifferentialKinematics(
            self.model,
            site="ee",  # both robot kinds label their TCP site "ee"
            joints=controlled,
            command_type=CommandType.POSE,
            ik_method=IkMethod.TRANSPOSE if args.transpose_solver else IkMethod.DAMPED_LEAST_SQUARES,
            bandwidth=BANDWIDTH,
        )
        assert self.controller.robot_count == ROBOT_COUNT

        self._input = self.controller.input()
        self._output = self.controller.output()
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        # Only the controlled DOFs are ever written, so the finger targets
        # seeded below survive untouched — no hand-rolled scatter needed.
        self._output.joint_target_q = self.control.joint_target_q

        # Seed every joint target with the home pose. The controlled slots are
        # overwritten each frame; the gripper slots hold the fingers open.
        wp.copy(self.control.joint_target_q, self.model.joint_q)

        # ---- Targets ---------------------------------------------------------
        # One step to populate the controller's site poses, which become the
        # initial targets so nothing jumps on the first frame.
        self._targets = np.zeros((ROBOT_COUNT, 7), dtype=np.float32)
        self._assign_targets_from_sites(warm_up=True)
        self.gizmo_tfs = [wp.transform(p=wp.vec3(*t[:3]), q=wp.quat(*t[3:])) for t in self._targets]

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.0, 0.0, 1.6), pitch=-15.0, yaw=180.0)

        # Graph capture is skipped: gizmo drags are pushed from Python each
        # frame, which is not capturable.
        self.graph = None

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _site_frames(self):
        """Return each controlled site's current world pose as (pos, quat) rows."""
        return self.controller._ee_pos.numpy().copy(), self.controller._ee_quat.numpy().copy()

    def _assign_targets_from_sites(self, warm_up: bool = False) -> None:
        """Seed the targets with the current site poses."""
        if warm_up:
            self._input.target_pos.zero_()
            self._input.target_quat.assign(np.tile([0.0, 0.0, 0.0, 1.0], (ROBOT_COUNT, 1)).astype(np.float32))
            self.controller.step(inputs=self._input, outputs=self._output, dt=0.0)
        pos, quat = self._site_frames()
        self._targets[:, :3] = pos
        self._targets[:, 3:] = quat
        self._push_targets()

    def _push_targets(self) -> None:
        self._input.target_pos.assign(np.ascontiguousarray(self._targets[:, :3]))
        self._input.target_quat.assign(np.ascontiguousarray(self._targets[:, 3:]))

    def _pull_gizmos(self) -> None:
        """Read the (possibly dragged) gizmo frames back into the target arrays."""
        for r, tf in enumerate(self.gizmo_tfs):
            p = wp.transform_get_translation(tf)
            q = wp.transform_get_rotation(tf)
            self._targets[r, :3] = [p[0], p[1], p[2]]
            self._targets[r, 3:] = [q[0], q[1], q[2], q[3]]
        self._push_targets()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._pull_gizmos()
        # Rebind to whichever State buffer the substep swap left at state_0.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd

        # One controller step per frame. The position target is one frame ahead;
        # the MuJoCo PD then tracks that fixed target for every substep. Running
        # IK inside the substep loop would refresh the target against the
        # drifted current q each substep, leaving the PD no restoring signal.
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.frame_dt)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        pos, quat = self._site_frames()
        for r, tf in enumerate(self.gizmo_tfs):
            snap = wp.transform(p=wp.vec3(*pos[r]), q=wp.quat(*quat[r]))
            self.viewer.log_gizmo(f"target_{r}", tf, snap_to=snap)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify every arm tracks its target and the excluded fingers stay put."""
        joint_q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"

        # The fleet is genuinely ragged: 7 controlled DOFs per Franka, 6 per UR10.
        assert self.controller.controlled_dof_count == FRANKA_COUNT * 7 + UR10_COUNT * 6
        assert self.controller.max_dofs == 7

        # Untouched gizmos leave the targets at the home site poses, so every
        # site should still be sitting on its target.
        pos, _ = self._site_frames()
        np.testing.assert_allclose(
            pos,
            self._targets[:, :3],
            atol=0.05,
            err_msg=f"sites drifted from their targets:\n{pos}\nvs\n{self._targets[:, :3]}",
        )

        # The gripper fingers are excluded from control, so the PD holds them at
        # the home width the seeding put there.
        art_start = self.model.articulation_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        for robot in range(FRANKA_COUNT):
            base = int(qd_start[art_start[robot]])
            fingers = joint_q[base + 7 : base + 9]
            np.testing.assert_allclose(
                fingers, FRANKA_HOME[7:9], atol=0.01, err_msg=f"robot {robot} fingers moved: {fingers}"
            )

        # Holding still says nothing about the IK, so displace every target and
        # verify the fleet actually converges on the new pose.
        self._targets[:, :3] += np.array([0.0, 0.0, -0.12], dtype=np.float32)
        self.gizmo_tfs = [wp.transform(p=wp.vec3(*t[:3]), q=wp.quat(*t[3:])) for t in self._targets]
        for _ in range(90):
            self.step()

        moved, _ = self._site_frames()
        np.testing.assert_allclose(
            moved,
            self._targets[:, :3],
            atol=0.06,
            err_msg=f"fleet did not track the displaced target:\n{moved}\nvs\n{self._targets[:, :3]}",
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
