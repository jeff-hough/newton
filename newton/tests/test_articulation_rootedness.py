# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repro: an articulation whose parent body belongs to a *different* articulation.

``eval_fk``, ``eval_jacobian``, and ``eval_mass_matrix`` all assume every
articulation is **rooted** — that each of its joints has either the world
(``-1``) or a body owned by a joint in the *same* articulation as its parent.
Nothing in Newton enforces or documents that assumption, and violating it
produces silently wrong dynamics in two independent ways:

1. ``eval_articulation_jacobian`` walks ``joint_ancestor`` out of the
   articulation and writes that ancestor's motion subspace at column
   ``joint_qd_start[ancestor] - articulation_dof_start``, which is negative.
   Deterministic; same wrong answer on every device.
2. ``eval_fk`` reads ``body_q[parent]`` for a body another articulation owns,
   which another thread is concurrently writing. Device-dependent.

Every model below is the same two-body mechanism. Only the articulation
grouping differs.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices

MASS = 2.0  # [kg]
ARM = 1.0  # offset mass' distance from its own joint [m]
OFFSET = 1.0  # pendulum joint's distance from the base joint [m]
IZZ = 0.1  # spin inertia about the offset mass' own COM [kg m^2]
INERTIA = wp.mat33(np.diag([IZZ, IZZ, IZZ]).astype(np.float32))
BASE_ANGLE = 0.7  # [rad]


def _offset_mass_link(builder):
    """A 2 kg mass on a massless 1 m arm. Inertia locked so finalize() cannot rewrite it."""
    return builder.add_link(mass=MASS, com=wp.vec3(ARM, 0.0, 0.0), inertia=INERTIA, lock_inertia=True)


def _add_base(builder):
    """Add the base body and its world-parented joint. Returns (body, joint)."""
    base = builder.add_link(mass=1.0, inertia=INERTIA, lock_inertia=True)
    joint = builder.add_joint_revolute(parent=-1, child=base, axis=wp.vec3(0.0, 0.0, 1.0))
    return base, joint


def _add_pendulum(builder, base):
    """Add the offset mass, hinged 1 m out along ``base``. Returns the joint."""
    link = _offset_mass_link(builder)
    return builder.add_joint_revolute(
        parent=base,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(OFFSET, 0.0, 0.0)),
    )


def _build_split(device):
    """The mechanism as two articulations — the pendulum's parent body is in "base"."""
    builder = newton.ModelBuilder()
    base, base_joint = _add_base(builder)
    builder.add_articulation([base_joint], label="base")
    builder.add_articulation([_add_pendulum(builder, base)], label="pendulum")
    return builder.finalize(device=device)


def _build_rooted(device):
    """The identical mechanism as one rooted articulation. This is the correct reference."""
    builder = newton.ModelBuilder()
    base, base_joint = _add_base(builder)
    builder.add_articulation([base_joint, _add_pendulum(builder, base)], label="whole")
    return builder.finalize(device=device)


def _build_pendulum_alone(device):
    """Just the offset mass on a world-parented joint: 1 m rotation radius, nothing else."""
    builder = newton.ModelBuilder()
    link = _offset_mass_link(builder)
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([joint], label="pendulum")
    return builder.finalize(device=device)


def _eval(model, base_angle):
    """Run FK at ``base_angle`` with the pendulum joint at zero. Returns (body positions, H)."""
    state = model.state()
    state.joint_q.assign(np.array([base_angle, 0.0][: model.joint_coord_count], dtype=np.float32))
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state.body_q.numpy()[:, :3], newton.eval_mass_matrix(model, state).numpy()


class TestArticulationRootedness(unittest.TestCase):
    def test_mass_matrix_matches_when_pendulum_is_mounted_on_another_articulation(self):
        r"""Verify a pendulum's mass matrix is the same mounted as it is standalone.

        Both models below, drawn at q=0, with (o) a revolute joint about z and
        (*) the 2 kg mass. The pendulum is identical in both: same mass, same
        Izz, same 1 m radius about its own joint::

            standalone                     mounted
            ----------                     -------
            j0                             j0            j1
            o--------------*               o-------------o-------------*
            |<--- 1 m ---->|               |<--- 1 m --->|<--- 1 m --->|
                  ARM                           OFFSET         ARM
                                           \___ "base" _/\_ "pendulum" _/

        A 2 kg mass on a 1 m arm has moment of inertia ``Izz + m*ARM^2 = 2.1``
        about its own joint axis. That is a property of the pendulum alone, so
        mounting it on another body cannot change it.

        Newton returns 8.1 = ``Izz + m*(ARM + OFFSET)^2``: the radius measured
        from j0 rather than j1, because the ancestor's Jacobian column overwrote
        the pendulum's own at a negative column index.
        """
        expected = IZZ + MASS * ARM**2
        for device in get_test_devices():
            with self.subTest(device=str(device)):
                _, h_alone = _eval(_build_pendulum_alone(device), 0.0)
                _, h_split = _eval(_build_split(device), 0.0)

                # pendulum on its own:
                self.assertAlmostEqual(float(h_alone[0, 0, 0]), expected, places=5, msg="reference is wrong")

                # pendulum on its own, as the second articulation:
                self.assertAlmostEqual(
                    float(h_split[1, 0, 0]),
                    expected,
                    places=5,
                    msg=f"mounted pendulum H={float(h_split[1, 0, 0])}, standalone H={expected}",
                )

    def test_body_poses_match_when_mechanism_split_across_articulations(self):
        r"""Verify body poses depend on the kinematic tree, not on articulation grouping.

        The base joint is turned to 0.7 rad and the pendulum joint left at 0, so
        the whole arm should swing rigidly. j1 rides on the base and must end up
        rotated with it::

            one articulation (and CPU)     split across two, on CUDA
            --------------------------     -------------------------
                      * mass
                     /
                j1  o                      j0  o-------o-------------* mass
                   /                           base    j1
                  / 0.7 rad
            j0   o------ base
                                           j1 at [1, 0, 0] — placed from an
            j1 at [0.765, 0.644, 0]        unrotated base; the arm never swung

        ``body_q`` is a function of ``joint_q`` and the parent/child topology.
        Splitting the same two joints across two articulations changes neither,
        so the poses must be identical.

        On CUDA the pendulum articulation reads ``body_q[base]`` before the base
        articulation has written it, and places the link as though the base were
        unrotated: ``[1, 0, 0]`` instead of ``[cos 0.7, sin 0.7, 0]``. CPU
        happens to schedule the base first and gets the right answer, so this
        test passes there.
        """
        for device in get_test_devices():
            with self.subTest(device=str(device)):
                rooted, _ = _eval(_build_rooted(device), BASE_ANGLE)
                split, _ = _eval(_build_split(device), BASE_ANGLE)
                np.testing.assert_allclose(
                    split,
                    rooted,
                    atol=1e-5,
                    err_msg="the same mechanism produced different body poses when split across articulations",
                )

    def test_non_rooted_articulation_is_rejected(self):
        r"""Verify building an articulation whose parent body is outside it raises.

        The topology both tests above rely on. j1's parent is a body owned by a
        different articulation, so "pendulum" has no root of its own::

            world ---o---> base ---o---> link
                     j0            j1
                     \___/         \____/
                    "base"       "pendulum"
                                     ^
                                     parent body (base) belongs to "base",
                                     so this articulation is not rooted

        The alternative fix to both defects above: if cross-articulation
        mounting is not meant to be supported, ``add_articulation`` should say
        so. It currently validates joint contiguity, world membership, and
        single-parent-within-the-articulation, but never rootedness.
        """
        builder = newton.ModelBuilder()
        base, base_joint = _add_base(builder)
        builder.add_articulation([base_joint], label="base")
        pendulum_joint = _add_pendulum(builder, base)
        with self.assertRaises(ValueError):
            builder.add_articulation([pendulum_joint], label="pendulum")


if __name__ == "__main__":
    unittest.main()
