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
carrying a 2 kg mass on a second revolute joint 1 m further out. One test adds a
bystander body on its own joint, to have something for a stray write to land in.
Each test spells out its own dimensions, so it can be read on its own.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices


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
    def test_jacobian_predicts_the_velocity_fk_computes_for_a_mounted_pendulum(self):
        r"""Verify ``J @ joint_qd`` reproduces the body velocity ``eval_fk`` computes.

        :func:`newton.eval_jacobian` documents that ``J_link @ joint_qd ==
        state.body_qd[link]``, so the two ways of asking "how fast is this body
        moving?" must agree. Only the pendulum joint spins, at 1 rad/s::

            j0         j1        mass
            o----------o--------->*   the COM is 1 m from j1,
            |<- 1 m -->|<- 1 m -->|   so it moves at 1 m/s
            \_"base"_/\_"pendulum"_/

        ``eval_fk`` gets 1 m/s. The Jacobian says 2 m/s — the speed the COM would
        have if it were spinning about j0, 2 m away — because the pendulum's
        ancestor is the base joint, which lies outside the pendulum articulation.
        Its column index comes out negative and lands on the pendulum's own
        column, overwriting it.
        """
        inertia = wp.mat33(np.diag([0.1, 0.1, 0.1]).astype(np.float32))  # about each body's own COM [kg m^2]
        axis = wp.vec3(0.0, 0.0, 1.0)

        for device in get_test_devices():
            with self.subTest(device=str(device)):
                builder = newton.ModelBuilder()
                base = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
                builder.add_articulation([builder.add_joint_revolute(parent=-1, child=base, axis=axis)], label="base")
                mass = builder.add_link(
                    mass=2.0,
                    com=wp.vec3(1.0, 0.0, 0.0),  # 1 m out from its own joint
                    inertia=inertia,
                    lock_inertia=True,
                )
                builder.add_articulation(
                    [
                        builder.add_joint_revolute(
                            parent=base,
                            child=mass,
                            axis=axis,
                            parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),  # 1 m out from the base body
                        )
                    ],
                    label="pendulum",
                )
                model = builder.finalize(device=device)

                state = model.state()
                state.joint_qd.assign(np.array([0.0, 1.0], dtype=np.float32))  # spin only the pendulum joint
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                pendulum_jacobian = newton.eval_jacobian(model, state).numpy()[1]
                predicted = pendulum_jacobian @ np.array([1.0])
                np.testing.assert_allclose(
                    predicted,
                    state.body_qd.numpy()[1],
                    atol=1e-5,
                    err_msg=(
                        "the Jacobian and forward kinematics disagree about the body's velocity\n"
                        f"  J @ joint_qd:   {np.array2string(predicted, precision=4)}\n"
                        f"  state.body_qd:  {np.array2string(state.body_qd.numpy()[1], precision=4)}"
                    ),
                )

    def test_jacobian_block_is_unchanged_by_an_unrelated_articulation(self):
        r"""Verify an articulation's Jacobian block does not depend on the others.

        Where the previous test shows the wrong number, this one shows the write
        leaving the memory it owns. ``eval_jacobian`` gives each articulation its
        own ``J[i]`` block, so the "spinner" block below is a function of that
        articulation alone. Adding a pendulum that shares no body, joint or DOF
        with it cannot change it::

            spinner       base            pendulum
            o             o----------o----------*
                          \_"base"_/\_"pendulum"_/

        The pendulum's ancestor is the base joint, outside its articulation, so
        its column comes out at ``joint_qd_start[ancestor] -
        articulation_dof_start = -2`` — two below its own single-column block.
        Warp resolves a negative index by adding the dimension size once, which
        here leaves it at ``-1``, so the write lands in the spinner's block and
        overwrites its last entry.

        No expected value is asserted, only that the block is untouched, so this
        holds whatever the Jacobian ought to contain.
        """
        inertia = wp.mat33(np.diag([0.1, 0.1, 0.1]).astype(np.float32))  # about each body's own COM [kg m^2]
        axis = wp.vec3(0.0, 0.0, 1.0)

        def spinner_jacobian_block(device, *, mount_pendulum):
            builder = newton.ModelBuilder()

            base = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
            builder.add_articulation([builder.add_joint_revolute(parent=-1, child=base, axis=axis)], label="base")

            spinner = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
            builder.add_articulation([builder.add_joint_revolute(parent=-1, child=spinner, axis=axis)], label="spinner")

            if mount_pendulum:
                mass = builder.add_link(
                    mass=2.0,
                    com=wp.vec3(1.0, 0.0, 0.0),  # 1 m out from its own joint
                    inertia=inertia,
                    lock_inertia=True,
                )
                builder.add_articulation(
                    [
                        builder.add_joint_revolute(
                            parent=base,
                            child=mass,
                            axis=axis,
                            parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),  # 1 m out from the base body
                        )
                    ],
                    label="pendulum",
                )

            model = builder.finalize(device=device)
            return newton.eval_jacobian(model, _fk(model)).numpy()[1]

        for device in get_test_devices():
            with self.subTest(device=str(device)):
                alone = spinner_jacobian_block(device, mount_pendulum=False)
                mounted = spinner_jacobian_block(device, mount_pendulum=True)
                np.testing.assert_array_equal(
                    mounted,
                    alone,
                    err_msg=(
                        "adding an unrelated articulation changed the spinner's Jacobian block\n"
                        f"  without the pendulum: {alone.ravel()}\n"
                        f"  with the pendulum:    {mounted.ravel()}"
                    ),
                )

    def test_mass_matrix_matches_when_pendulum_is_mounted_on_another_articulation(self):
        r"""Verify a pendulum's mass matrix is the same mounted as it is standalone.

        Both models drawn at q=0, with (o) a revolute joint about z and (*) the
        2 kg mass. The pendulum is identical in both: same mass, same Izz, same
        1 m radius about its own joint::

            standalone      mounted
            ----------      -------
            j0    mass      j0         j1        mass
            o----------*    o----------o----------*
            |<- 1 m -->|    |<- 1 m -->|<- 1 m -->|
                            \_"base"_/\_"pendulum"_/

        """

        inertia = wp.mat33(np.diag([0.1, 0.1, 0.1]).astype(np.float32))  # about each body's own COM [kg m^2]
        axis = wp.vec3(0.0, 0.0, 1.0)
        expected = 0.1 + 2.0 * 1.0**2  # Izz + m*r^2 about the pendulum's own joint

        for device in get_test_devices():
            with self.subTest(device=str(device)):
                alone = newton.ModelBuilder()
                mass = alone.add_link(mass=2.0, com=wp.vec3(1.0, 0.0, 0.0), inertia=inertia, lock_inertia=True)
                alone.add_articulation([alone.add_joint_revolute(parent=-1, child=mass, axis=axis)], label="pendulum")

                mounted = newton.ModelBuilder()
                base = mounted.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
                mass = mounted.add_link(mass=2.0, com=wp.vec3(1.0, 0.0, 0.0), inertia=inertia, lock_inertia=True)
                mounted.add_articulation([mounted.add_joint_revolute(parent=-1, child=base, axis=axis)], label="base")
                hanging_joint = mounted.add_joint_revolute(
                    parent=base,
                    child=mass,
                    axis=axis,
                    parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),  # 1 m out from the base body
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
        lands 1 m from the world origin at ``[1, 0, 0]``. Note that is not the
        base's q=0 pose either, which would put it at ``[2, 0, 0]``. CPU happens
        to schedule the base first, so this test passes there.
        """
        inertia = wp.mat33(np.diag([0.1, 0.1, 0.1]).astype(np.float32))  # about each body's own COM [kg m^2]
        axis = wp.vec3(0.0, 0.0, 1.0)

        def build(device, *, split):
            builder = newton.ModelBuilder()
            base = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
            mass = builder.add_link(mass=2.0, com=wp.vec3(1.0, 0.0, 0.0), inertia=inertia, lock_inertia=True)
            base_joint = builder.add_joint_revolute(
                parent=-1,
                child=base,
                axis=axis,
                child_xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)),  # base origin sits 1 m out from its axis
            )
            hanging_joint = builder.add_joint_revolute(
                parent=base,
                child=mass,
                axis=axis,
                parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),  # 1 m out from the base body
            )
            if split:
                builder.add_articulation([base_joint], label="base")
                builder.add_articulation([hanging_joint], label="pendulum")
            else:
                builder.add_articulation([base_joint, hanging_joint], label="whole")
            return builder.finalize(device=device)

        for device in get_test_devices():
            with self.subTest(device=str(device)):
                whole = _fk(build(device, split=False), 0.7).body_q.numpy()[:, :3]
                apart = _fk(build(device, split=True), 0.7).body_q.numpy()[:, :3]
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
        with — landing it 1 m from the world origin at ``[1, 0, 0]``. Note that
        is not the base's q=0 pose either, which would put it at ``[2, 0, 0]``.

        No second articulation is involved here. One articulation whose joints
        are not in topological order is enough, and ``add_articulation``
        accepts it.
        """
        inertia = wp.mat33(np.diag([0.1, 0.1, 0.1]).astype(np.float32))  # about each body's own COM [kg m^2]
        axis = wp.vec3(0.0, 0.0, 1.0)

        def build(device, *, base_joint_first):
            builder = newton.ModelBuilder()
            base = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
            mass = builder.add_link(mass=2.0, com=wp.vec3(1.0, 0.0, 0.0), inertia=inertia, lock_inertia=True)

            def add_base_joint():
                return builder.add_joint_revolute(
                    parent=-1,
                    child=base,
                    axis=axis,
                    child_xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)),  # base origin sits 1 m out from its axis
                )

            def add_hanging_joint():
                return builder.add_joint_revolute(
                    parent=base,
                    child=mass,
                    axis=axis,
                    parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),  # 1 m out from the base body
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
                expected = _fk(ordered, 0.7, ordered_base).body_q.numpy()[:, :3]
                actual = _fk(shuffled, 0.7, shuffled_base).body_q.numpy()[:, :3]
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


if __name__ == "__main__":
    unittest.main()
