# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repro: ``eval_fk`` and ``eval_jacobian`` assume articulations are rooted and ordered.

Both walk an articulation's joints in index order and expect each joint's parent
body to have been placed already. That holds only if every joint's parent is
either the world (``-1``) or a body whose owning joint is in the *same*
articulation and at a *lower* index. Nothing in Newton enforces or documents it,
and violating it produces silently wrong dynamics two ways:

1. ``eval_articulation_jacobian`` walks ``joint_ancestor`` out of the
   articulation and writes that ancestor's motion subspace at column
   ``joint_qd_start[ancestor] - articulation_dof_start``, which is negative.
2. ``eval_fk`` reads a ``body_q[parent]`` that has not been written yet — by
   another articulation's thread, or by a later joint in its own articulation.

Every test builds the same two-body mechanism: a base body on a revolute joint,
carrying a 2 kg mass on a second revolute joint 1 m further out.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices

MASS = 2.0  # mass of the carried body [kg]
ARM = 1.0  # its COM offset from its own joint [m]
OFFSET = 1.0  # its joint's offset from the base body origin [m]
BASE_ARM = 1.0  # base body origin's offset from its own joint axis [m]
IZZ = 0.1  # spin inertia of each body about its own COM [kg m^2]
INERTIA = wp.mat33(np.diag([IZZ, IZZ, IZZ]).astype(np.float32))
BASE_ANGLE = 0.7  # [rad]
AXIS = wp.vec3(0.0, 0.0, 1.0)


def _fk(model, base_angle=0.0, base_joint=0):
    """Run forward kinematics with ``base_joint`` at ``base_angle``, all else at zero."""
    joint_q = np.zeros(model.joint_coord_count, dtype=np.float32)
    joint_q[base_joint] = base_angle
    state = model.state()
    state.joint_q.assign(joint_q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


class TestArticulationRootedness(unittest.TestCase):
    def test_mass_matrix_matches_when_pendulum_is_mounted_on_another_articulation(self):
        r"""Verify a pendulum's mass matrix is the same mounted as it is standalone.

        Both models drawn at q=0, with (o) a revolute joint about z and (*) the
        2 kg mass. The pendulum is identical in both: same mass, same Izz, same
        ARM = 1 m radius about its own joint::

            standalone      mounted
            ----------      -------
            j0    mass      j0         j1        mass
            o----------*    o----------o----------*
            |<- 1 m -->|    |<- 1 m -->|<- 1 m -->|
                ARM            OFFSET       ARM
                            \_"base"_/\_"pendulum"_/

        A 2 kg mass on a 1 m arm has moment of inertia ``Izz + m*ARM^2 = 2.1``
        about its own joint axis. That is a property of the pendulum alone, so
        mounting it on another body cannot change it.

        Newton returns 8.1 = ``Izz + m*(OFFSET + ARM)^2`` — the radius measured
        from j0 rather than j1, because the ancestor's Jacobian column overwrote
        the pendulum's own at a negative column index.

        The base body sits on its own joint axis here, which keeps the result
        deterministic. Offsetting it would let the FK race of the other tests
        feed this number too, making it differ per device.
        """
        expected = IZZ + MASS * ARM**2
        for device in get_test_devices():
            with self.subTest(device=str(device)):
                alone = newton.ModelBuilder()
                mass = alone.add_link(mass=MASS, com=wp.vec3(ARM, 0.0, 0.0), inertia=INERTIA, lock_inertia=True)
                alone.add_articulation([alone.add_joint_revolute(parent=-1, child=mass, axis=AXIS)], label="pendulum")

                mounted = newton.ModelBuilder()
                base = mounted.add_link(mass=1.0, inertia=INERTIA, lock_inertia=True)
                mass = mounted.add_link(mass=MASS, com=wp.vec3(ARM, 0.0, 0.0), inertia=INERTIA, lock_inertia=True)
                mounted.add_articulation([mounted.add_joint_revolute(parent=-1, child=base, axis=AXIS)], label="base")
                hanging_joint = mounted.add_joint_revolute(
                    parent=base, child=mass, axis=AXIS, parent_xform=wp.transform(p=wp.vec3(OFFSET, 0.0, 0.0))
                )
                mounted.add_articulation([hanging_joint], label="pendulum")

                alone = alone.finalize(device=device)
                mounted = mounted.finalize(device=device)
                h_alone = float(newton.eval_mass_matrix(alone, _fk(alone)).numpy()[0, 0, 0])
                h_mounted = float(newton.eval_mass_matrix(mounted, _fk(mounted)).numpy()[1, 0, 0])

                self.assertAlmostEqual(h_alone, expected, places=5, msg="reference is wrong")
                self.assertAlmostEqual(
                    h_mounted,
                    expected,
                    places=5,
                    msg=f"mounted pendulum H={h_mounted}, standalone H={expected}",
                )

    def test_body_poses_match_when_mechanism_split_across_articulations(self):
        r"""Verify body poses do not depend on how joints are grouped into articulations.

        The base joint is turned to 0.7 rad and the hanging joint left at 0, so
        the whole arm should swing rigidly::

            one articulation (and CPU)   split across two, on CUDA
            --------------------------   -------------------------
                     * mass
                    /                          + base [0.765, 0.644]
               j1  o                          /  (correct in both)
                  /                          /
            base +                          /
                /                          /
            j0 o                    j0    o---------------* mass [1, 0, 0]

            base at [0.765, 0.644]       base is right, the mass is not:
            mass at [1.530, 1.288]       the link was laid out from the
                                         world origin

        ``body_q`` is a function of ``joint_q`` and the parent/child topology.
        Splitting the same two joints across two articulations changes neither,
        so the poses must be identical.

        On CUDA the pendulum articulation reads ``body_q[base]`` before the base
        articulation has written it. What it reads is the *unwritten* value — the
        identity transform ``state.body_q`` was allocated with — so the link
        lands OFFSET from the world origin at ``[1, 0, 0]``. Note that is not
        the base's q=0 pose either, which would put it at ``[2, 0, 0]``. CPU
        happens to schedule the base first, so this test passes there.
        """

        def build(device, *, split):
            builder = newton.ModelBuilder()
            base = builder.add_link(mass=1.0, inertia=INERTIA, lock_inertia=True)
            mass = builder.add_link(mass=MASS, com=wp.vec3(ARM, 0.0, 0.0), inertia=INERTIA, lock_inertia=True)
            base_joint = builder.add_joint_revolute(
                parent=-1, child=base, axis=AXIS, child_xform=wp.transform(p=wp.vec3(-BASE_ARM, 0.0, 0.0))
            )
            hanging_joint = builder.add_joint_revolute(
                parent=base, child=mass, axis=AXIS, parent_xform=wp.transform(p=wp.vec3(OFFSET, 0.0, 0.0))
            )
            if split:
                builder.add_articulation([base_joint], label="base")
                builder.add_articulation([hanging_joint], label="pendulum")
            else:
                builder.add_articulation([base_joint, hanging_joint], label="whole")
            return builder.finalize(device=device)

        for device in get_test_devices():
            with self.subTest(device=str(device)):
                whole = _fk(build(device, split=False), BASE_ANGLE).body_q.numpy()[:, :3]
                apart = _fk(build(device, split=True), BASE_ANGLE).body_q.numpy()[:, :3]
                np.testing.assert_allclose(
                    apart,
                    whole,
                    atol=1e-5,
                    err_msg=(
                        "the same mechanism produced different body poses when split across articulations\n"
                        f"  base body: one articulation={np.array2string(whole[0], precision=4)}"
                        f"  split={np.array2string(apart[0], precision=4)}\n"
                        f"  mass body: one articulation={np.array2string(whole[1], precision=4)}"
                        f"  split={np.array2string(apart[1], precision=4)}"
                    ),
                )

    def test_body_poses_match_when_joints_are_declared_in_a_different_order(self):
        r"""Verify body poses do not depend on the order the joints were declared.

        The two models are built with **identical topology**: the same two
        bodies, the same parent/child link between them, the same joint types
        and joint frames, all in a single articulation. The only difference is
        which joint was created first, which changes nothing physical — only
        where each joint lands in ``joint_q``::

            declared base-first           declared hanging-first
            -------------------           ----------------------
            joint 0 = base                joint 0 = hanging
            joint 1 = hanging             joint 1 = base

        Each model is then placed in the same physical configuration, indexing
        through its *own* base-joint slot: base joint at 0.7 rad, hanging joint
        at 0. Body poses are a function of the topology and the configuration
        alone — declaration order is not an input to forward kinematics — and
        both of those now match, so the bodies must end up in the same place.

        The base does rotate correctly in both; only the mass body differs::

            correct                      what Newton computes
            -------                      --------------------
                     * mass
                    /                          + base [0.765, 0.644]
            hanging o                         /  (correct in both)
                   /                         /
             base +                         /
                 /                         /
            base o                  base  o---------------* mass [1, 0, 0]
             joint                  joint

            base at [0.765, 0.644]       base is right, the mass is not:
            mass at [1.530, 1.288]       the link was laid out from the
                                         world origin

        ``eval_fk`` walks an articulation's joints in index order and expects
        each joint's parent body to have been placed already. Declaring the
        hanging joint first breaks that: the link is positioned from the
        *unwritten* base transform — the identity ``state.body_q`` was allocated
        with — landing it OFFSET from the world origin at ``[1, 0, 0]``. Note
        that is not the base's q=0 pose either, which would put it at
        ``[2, 0, 0]``.

        No second articulation is involved here. One articulation whose joints
        are not in topological order is enough, and ``add_articulation``
        accepts it.
        """

        def build(device, *, base_joint_first):
            builder = newton.ModelBuilder()
            base = builder.add_link(mass=1.0, inertia=INERTIA, lock_inertia=True)
            mass = builder.add_link(mass=MASS, com=wp.vec3(ARM, 0.0, 0.0), inertia=INERTIA, lock_inertia=True)

            def add_base_joint():
                return builder.add_joint_revolute(
                    parent=-1, child=base, axis=AXIS, child_xform=wp.transform(p=wp.vec3(-BASE_ARM, 0.0, 0.0))
                )

            def add_hanging_joint():
                return builder.add_joint_revolute(
                    parent=base, child=mass, axis=AXIS, parent_xform=wp.transform(p=wp.vec3(OFFSET, 0.0, 0.0))
                )

            if base_joint_first:
                base_joint, hanging_joint = add_base_joint(), add_hanging_joint()
            else:
                hanging_joint, base_joint = add_hanging_joint(), add_base_joint()
            builder.add_articulation(sorted([base_joint, hanging_joint]), label="chain")
            return builder.finalize(device=device), base_joint

        for device in get_test_devices():
            with self.subTest(device=str(device)):
                ordered, ordered_base = build(device, base_joint_first=True)
                shuffled, shuffled_base = build(device, base_joint_first=False)
                expected = _fk(ordered, BASE_ANGLE, ordered_base).body_q.numpy()[:, :3]
                actual = _fk(shuffled, BASE_ANGLE, shuffled_base).body_q.numpy()[:, :3]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-5,
                    err_msg=(
                        "declaring the joints in a different order moved the bodies\n"
                        f"  base body: base-first={np.array2string(expected[0], precision=4)}"
                        f"  hanging-first={np.array2string(actual[0], precision=4)}\n"
                        f"  mass body: base-first={np.array2string(expected[1], precision=4)}"
                        f"  hanging-first={np.array2string(actual[1], precision=4)}"
                    ),
                )

    def test_non_rooted_articulation_is_rejected(self):
        r"""Verify building an articulation whose parent body is outside it raises.

        The topology the first two tests rely on. The hanging joint's parent is
        a body owned by a different articulation, so "pendulum" has no root::

            world ---o---> base ---o---> mass
                     j0            j1
                     \___/         \____/
                    "base"       "pendulum"
                                     ^
                                     parent body (base) belongs to "base",
                                     so this articulation is not rooted

        The alternative fix: if cross-articulation mounting is not meant to be
        supported, ``add_articulation`` should say so. It currently validates
        joint contiguity, world membership, and single-parent-within-the-
        articulation, but never rootedness.
        """
        builder = newton.ModelBuilder()
        base = builder.add_link(mass=1.0, inertia=INERTIA, lock_inertia=True)
        mass = builder.add_link(mass=MASS, com=wp.vec3(ARM, 0.0, 0.0), inertia=INERTIA, lock_inertia=True)
        builder.add_articulation([builder.add_joint_revolute(parent=-1, child=base, axis=AXIS)], label="base")
        hanging_joint = builder.add_joint_revolute(
            parent=base, child=mass, axis=AXIS, parent_xform=wp.transform(p=wp.vec3(OFFSET, 0.0, 0.0))
        )
        with self.assertRaises(ValueError):
            builder.add_articulation([hanging_joint], label="pendulum")


if __name__ == "__main__":
    unittest.main()
