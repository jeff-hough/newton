# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the joint-impedance controller's DOF index derivation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.controllers.dof_indices import _build_dof_indices
from newton.controllers import select_joints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(labels, articulation_label=None, floating_base=False):
    """Build a revolute chain as a standalone builder, one articulation.

    Args:
        labels: Joint label per link, or ``None`` entries for default labels.
        articulation_label: Label for the articulation, or ``None``.
        floating_base: Prepend a 6-DOF free joint, so coordinate and DOF
            indices diverge downstream of it.
    """
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    if floating_base:
        torso = builder.add_link()
        joints.append(builder.add_joint_free(child=torso))
        builder.add_shape_box(body=torso, hx=0.2, hy=0.1, hz=0.1)
        prev = torso
    for label in labels:
        link = builder.add_link()
        kwargs = {"label": label} if label is not None else {}
        joints.append(builder.add_joint_revolute(parent=prev, child=link, axis=wp.vec3(0.0, 0.0, 1.0), **kwargs))
        builder.add_shape_box(body=link, hx=0.02, hy=0.1, hz=0.02)
        prev = link
    builder.add_articulation(joints, label=articulation_label)
    return builder


def _arm(**kwargs):
    """Build a standalone 3-link arm labelled shoulder/elbow/wrist."""
    return _chain(["shoulder", "elbow", "wrist"], articulation_label="arm", **kwargs)


def _leg():
    """Build a standalone 2-link leg labelled hip/knee."""
    return _chain(["hip", "knee"], articulation_label="leg")


def _with_loop_closure():
    """Finalize a scene whose first articulation is followed by a loop-closing joint.

    Loop-closing joints may not belong to an articulation, so they sit between
    ``articulation_end[i]`` and ``articulation_start[i + 1]``. They are ordinary
    revolute joints, so only that bound excludes them.
    """
    builder = newton.ModelBuilder()
    links = [builder.add_link() for _ in range(3)]
    for link in links:
        builder.add_shape_box(body=link, hx=0.02, hy=0.1, hz=0.02)
    chain = [
        builder.add_joint_revolute(parent=-1, child=links[0], axis=wp.vec3(0.0, 0.0, 1.0), label="j0"),
        builder.add_joint_revolute(parent=links[0], child=links[1], axis=wp.vec3(0.0, 0.0, 1.0), label="j1"),
        builder.add_joint_revolute(parent=links[1], child=links[2], axis=wp.vec3(0.0, 0.0, 1.0), label="j2"),
    ]
    builder.add_articulation(chain, label="loop_arm")
    builder.add_joint_revolute(parent=links[2], child=links[0], axis=wp.vec3(0.0, 0.0, 1.0), label="loop")

    solo = builder.add_link()
    builder.add_shape_box(body=solo, hx=0.02, hy=0.1, hz=0.02)
    builder.add_articulation(
        [builder.add_joint_revolute(parent=-1, child=solo, axis=wp.vec3(0.0, 0.0, 1.0), label="solo")],
        label="solo_arm",
    )
    return builder.finalize()


def _fleet(*specs):
    """Finalize a scene replicating each ``(builder, count)`` spec in order."""
    scene = newton.ModelBuilder()
    for builder, count in specs:
        scene.replicate(builder, world_count=count, spacing=(1.0, 0.0, 0.0))
    return scene.finalize()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDofIndices(unittest.TestCase):
    def test_selects_all_scalar_joints_by_default(self):
        """Verify that omitting both selectors controls every scalar joint."""
        model = _fleet((_arm(), 2))
        idx = _build_dof_indices(model)

        self.assertEqual(idx.robot_count, 2)
        self.assertEqual(idx.controlled_dof_count, 6)
        self.assertEqual(idx.max_dofs, 3)
        np.testing.assert_array_equal(idx.dof_offsets.numpy(), [0, 3, 6])
        np.testing.assert_array_equal(idx.selected_articulations.numpy(), [0, 1])

    def test_free_joint_diverges_coord_and_dof_indices(self):
        """Verify that coordinate and DOF indices differ downstream of a free joint.

        A free joint spans 7 coordinates but 6 DOFs, so the arm behind it sits at
        coordinates 7..9 while occupying DOFs 6..8. Conflating the two silently
        reads the wrong joint.
        """
        model = _fleet((_arm(floating_base=True), 1))
        self.assertNotEqual(model.joint_coord_count, model.joint_dof_count)

        idx = _build_dof_indices(model)

        np.testing.assert_array_equal(idx.controlled_dof_to_model_dof.numpy(), [6, 7, 8])
        np.testing.assert_array_equal(idx.controlled_dof_to_model_coord.numpy(), [7, 8, 9])
        np.testing.assert_array_equal(idx.articulation_dof_start.numpy(), [0])
        self.assertEqual(idx.controlled_joints, (1, 2, 3))

    def test_free_joint_is_read_but_not_controlled(self):
        """Verify that an uncontrolled free joint is excluded rather than rejected."""
        model = _fleet((_arm(floating_base=True), 1))
        idx = _build_dof_indices(model)

        # The free joint is joint 0; only the three revolute joints are controlled.
        self.assertEqual(idx.controlled_dof_count, 3)
        self.assertNotIn(0, idx.controlled_joints)

    def test_non_contiguous_joint_subset(self):
        """Verify index expansion when the selection skips joints within an articulation."""
        model = _fleet((_arm(floating_base=True), 1))
        idx = _build_dof_indices(model, joints=[1, 3])

        np.testing.assert_array_equal(idx.controlled_dof_to_model_dof.numpy(), [6, 8])
        np.testing.assert_array_equal(idx.controlled_dof_to_model_coord.numpy(), [7, 9])
        np.testing.assert_array_equal(idx.dof_offsets.numpy(), [0, 2])

    def test_label_broadcasts_across_a_fleet(self):
        """Verify that a joint label selects the matching joint of every robot."""
        model = _fleet((_arm(), 4))
        idx = _build_dof_indices(model, joints=["shoulder", "wrist"])

        self.assertEqual(idx.robot_count, 4)
        self.assertEqual(idx.controlled_dof_count, 8)
        self.assertEqual(idx.controlled_joints, (0, 2, 3, 5, 6, 8, 9, 11))

    def test_ordering_is_by_articulation_then_joint(self):
        """Verify that selection order does not affect the resulting DOF layout.

        Per-DOF gains are laid out robot-major, so the layout must depend only on
        the model, never on the order the caller happened to list joints in.
        """
        model = _fleet((_arm(), 2))
        forward = _build_dof_indices(model, joints=["shoulder", "wrist"])
        reversed_ = _build_dof_indices(model, joints=["wrist", "shoulder"])

        self.assertEqual(forward.controlled_joints, reversed_.controlled_joints)
        np.testing.assert_array_equal(
            forward.controlled_dof_to_model_dof.numpy(),
            reversed_.controlled_dof_to_model_dof.numpy(),
        )

    def test_heterogeneous_fleet_selected_by_label(self):
        """Verify that labels separate robot types with differing DOF counts."""
        model = _fleet((_arm(), 3), (_leg(), 2))

        arms = _build_dof_indices(model, joints=["shoulder", "elbow", "wrist"])
        self.assertEqual(arms.robot_count, 3)
        np.testing.assert_array_equal(arms.selected_articulations.numpy(), [0, 1, 2])

        legs = _build_dof_indices(model, joints=["hip", "knee"])
        self.assertEqual(legs.robot_count, 2)
        np.testing.assert_array_equal(legs.selected_articulations.numpy(), [3, 4])

    def test_heterogeneous_selection_is_ragged_not_padded(self):
        """Verify that robots with differing DOF counts produce ragged offsets."""
        model = _fleet((_arm(), 1), (_leg(), 1))
        idx = _build_dof_indices(model)

        np.testing.assert_array_equal(idx.dof_offsets.numpy(), [0, 3, 5])
        self.assertEqual(idx.controlled_dof_count, 5)
        self.assertEqual(idx.max_dofs, 3)

    def test_articulation_dof_start_tracks_each_robot(self):
        """Verify that each robot's articulation DOF start is recorded independently."""
        model = _fleet((_arm(floating_base=True), 2))
        idx = _build_dof_indices(model)

        # Each replica spans 9 DOFs (6 free + 3 revolute).
        np.testing.assert_array_equal(idx.articulation_dof_start.numpy(), [0, 9])
        np.testing.assert_array_equal(idx.controlled_dof_to_model_dof.numpy(), [6, 7, 8, 15, 16, 17])

    def test_excludes_loop_closing_joints(self):
        """Verify that a loop-closing joint is not controlled by a select-all.

        Such a joint lies inside the articulation's outer bound but past
        ``articulation_end``, and is an ordinary revolute joint, so bounding the
        scope by ``articulation_start[i + 1]`` would wrongly include it.
        """
        model = _with_loop_closure()
        self.assertEqual(model.joint_label[3], "loop")

        idx = _build_dof_indices(model)

        self.assertEqual(idx.controlled_joints, (0, 1, 2, 4))
        self.assertNotIn(3, idx.controlled_joints)
        np.testing.assert_array_equal(idx.dof_offsets.numpy(), [0, 3, 4])

    def test_rejects_explicitly_selected_loop_closing_joint(self):
        """Verify that naming a loop-closing joint fails rather than being silently dropped."""
        model = _with_loop_closure()
        with self.assertRaisesRegex(ValueError, "regular tree joints of the selected"):
            _build_dof_indices(model, joints=["loop"])

    def test_articulations_scope_a_label_selection(self):
        """Verify that articulations narrows which robots a joint label matches."""
        model = _fleet((_arm(), 4))
        idx = _build_dof_indices(model, articulations=[1, 3], joints=["elbow"])

        self.assertEqual(idx.robot_count, 2)
        np.testing.assert_array_equal(idx.selected_articulations.numpy(), [1, 3])
        self.assertEqual(idx.controlled_joints, (4, 10))

    def test_articulation_label_selects_every_instance(self):
        """Verify that an articulation label matches all replicas carrying it."""
        model = _fleet((_arm(), 3), (_leg(), 2))
        idx = _build_dof_indices(model, articulations=["leg"])

        self.assertEqual(idx.robot_count, 2)
        self.assertEqual(idx.controlled_dof_count, 4)


class TestDofIndicesErrors(unittest.TestCase):
    def test_rejects_non_scalar_controlled_joint(self):
        """Verify that explicitly selecting a free joint is rejected."""
        model = _fleet((_arm(floating_base=True), 1))
        with self.assertRaisesRegex(ValueError, "1-DOF revolute or prismatic"):
            _build_dof_indices(model, joints=[0])

    def test_rejects_unmatched_joint_label(self):
        """Verify that a mistyped joint label fails instead of selecting nothing."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "matches no joint"):
            _build_dof_indices(model, joints=["wrst"])

    def test_rejects_unmatched_articulation_label(self):
        """Verify that a mistyped articulation label fails loudly."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "matches no articulation"):
            _build_dof_indices(model, articulations=["leg"])

    def test_shared_label_matches_every_robot_kind_carrying_it(self):
        """Verify a label shared by two robot kinds selects the joints of both.

        Matching is exact string equality and always takes everything carrying
        the label. ModelBuilder numbers unlabelled joints per sub-builder, so
        two unrelated kinds both contain ``joint_1``; selecting it takes all four.
        """
        model = _fleet((_chain([None] * 3), 2), (_chain([None] * 2), 2))
        idx = _build_dof_indices(model, joints=["joint_1"])

        self.assertEqual(idx.robot_count, 4)
        self.assertEqual(idx.controlled_dof_count, 4)

    def test_articulations_narrow_a_shared_label(self):
        """Verify scoping by articulation restricts which robots a shared label reaches."""
        model = _fleet((_chain([None] * 3), 2), (_chain([None] * 2), 2))
        idx = _build_dof_indices(model, articulations=[0, 1], joints=["joint_1"])

        self.assertEqual(idx.robot_count, 2)
        self.assertEqual(idx.controlled_dof_count, 2)

    def test_same_label_across_identical_robots_is_not_ambiguous(self):
        """Verify that a label repeated across identical robots stays usable."""
        model = _fleet((_chain([None] * 3), 4))
        idx = _build_dof_indices(model, joints=["joint_1"])

        self.assertEqual(idx.robot_count, 4)

    def test_rejects_joint_outside_selected_articulations(self):
        """Verify that a joint index outside the articulation scope is rejected."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "not a regular tree joint of the selected"):
            _build_dof_indices(model, articulations=[0], joints=[4])

    def test_rejects_out_of_range_articulation_index(self):
        """Verify that an articulation index beyond the model is rejected."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "out of range"):
            _build_dof_indices(model, articulations=[7])

    def test_rejects_repeated_joint_selection(self):
        """Verify that selecting one joint twice is rejected.

        Two controlled DOFs mapping to one output slot would race in the scatter.
        """
        model = _fleet((_arm(), 1))
        with self.assertRaisesRegex(ValueError, "more than once"):
            _build_dof_indices(model, joints=[0, 1, 0])

    def test_rejects_repeated_articulation_selection(self):
        """Verify that selecting one articulation twice is rejected."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "more than once"):
            _build_dof_indices(model, articulations=[1, 1])

    def test_rejects_empty_selection_lists(self):
        """Verify that empty selectors are rejected rather than read as select-all."""
        model = _fleet((_arm(), 1))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _build_dof_indices(model, joints=[])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _build_dof_indices(model, articulations=[])


class TestSelectJoints(unittest.TestCase):
    def test_returns_indices_for_the_controller(self):
        """Verify a plain selection returns sorted model joint indices."""
        model = _fleet((_arm(), 2))
        self.assertEqual(select_joints(model), [0, 1, 2, 3, 4, 5])
        self.assertEqual(select_joints(model, articulations=[1]), [3, 4, 5])

    def test_exclude_drops_a_joint_from_every_robot(self):
        """Verify one excluded label covers a whole replicated fleet.

        This is the case whole-articulation selection cannot express: every robot
        is controlled except one joint inside each of them.
        """
        model = _fleet((_arm(), 3))  # shoulder, elbow, wrist per robot
        self.assertEqual(select_joints(model, exclude_joints=["wrist"]), [0, 1, 3, 4, 6, 7])

    def test_exclude_composes_with_both_selectors(self):
        """Verify exclusion applies after articulation and joint selection."""
        model = _fleet((_arm(), 3))
        selected = select_joints(model, articulations=[0, 2], joints=["shoulder", "wrist"], exclude_joints=["wrist"])
        self.assertEqual(selected, [0, 6])

    def test_result_drives_the_controller_selection(self):
        """Verify the returned list reproduces the same selection when passed back in."""
        model = _fleet((_arm(), 2))
        joints = select_joints(model, exclude_joints=["elbow"])
        idx = _build_dof_indices(model, joints=joints)
        self.assertEqual(idx.controlled_joints, tuple(joints))
        self.assertEqual(idx.robot_count, 2)

    def test_heterogeneous_fleet_exclusion(self):
        """Verify exclusion only touches robot types that carry the label."""
        model = _fleet((_arm(), 2), (_leg(), 2))
        selected = select_joints(model, exclude_joints=["wrist"])
        # The legs have no wrist, so they are untouched.
        self.assertEqual(selected, [0, 1, 3, 4, 6, 7, 8, 9])

    def test_exclude_articulations_drops_a_whole_robot_kind(self):
        """Verify one label removes every replica of a robot kind.

        Listing the kinds you keep scales with the fleet; excluding the one you
        do not want costs a single string however many kinds there are.
        """
        model = _fleet((_arm(), 2), (_leg(), 2))
        selected = select_joints(model, exclude_articulations=["leg"])
        self.assertEqual(selected, [0, 1, 2, 3, 4, 5])  # the two arms only

    def test_exclude_articulations_accepts_indices(self):
        """Verify articulation exclusion works by index as well as by label."""
        model = _fleet((_arm(), 3))
        self.assertEqual(select_joints(model, exclude_articulations=[1]), [0, 1, 2, 6, 7, 8])

    def test_exclusions_compose_across_both_levels(self):
        """Verify articulations are excluded before joints are resolved."""
        model = _fleet((_arm(), 2), (_leg(), 2))
        selected = select_joints(model, exclude_articulations=["leg"], exclude_joints=["elbow"])
        self.assertEqual(selected, [0, 2, 3, 5])  # arms only, without their elbows

    def test_rejects_articulation_exclusion_that_removes_nothing(self):
        """Verify an exclusion outside the selection fails rather than doing nothing."""
        model = _fleet((_arm(), 2), (_leg(), 2))
        with self.assertRaisesRegex(ValueError, "removed no articulation"):
            select_joints(model, articulations=["arm"], exclude_articulations=["leg"])

    def test_rejects_articulation_exclusion_that_removes_everything(self):
        """Verify excluding the entire selection is rejected."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "removed every selected articulation"):
            select_joints(model, exclude_articulations=["arm"])

    def test_rejects_unknown_articulation_exclusion(self):
        """Verify a mistyped articulation label is caught."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "matches no articulation"):
            select_joints(model, exclude_articulations=["leg"])

    def test_excluding_articulations_narrows_which_robots_a_label_reaches(self):
        """Verify dropping a robot kind removes it from a shared label's matches."""
        model = _fleet((_chain([None] * 3), 2), (_chain([None] * 2), 2))
        self.assertEqual(len(select_joints(model, joints=["joint_1"])), 4)  # all four robots
        self.assertEqual(select_joints(model, exclude_articulations=[2, 3], joints=["joint_1"]), [0, 3])

    def test_exclusion_reaches_every_robot_kind_sharing_the_label(self):
        """Verify one exclusion drops a joint from every kind that carries the label.

        This is the case a shared-label rejection used to block: two robot kinds
        with different layouts both having a ``gripper``, where dropping all of
        them is exactly the intent.
        """
        model = _fleet((_chain(["shoulder", "elbow", "gripper"]), 2), (_chain(["boom", "gripper"]), 2))
        self.assertEqual(select_joints(model, exclude_joints=["gripper"]), [0, 1, 3, 4, 6, 8])

    def test_rejects_exclusion_matching_nothing(self):
        """Verify a mistyped exclusion fails instead of silently controlling the joint."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "matches no joint"):
            select_joints(model, exclude_joints=["wrst"])

    def test_rejects_exclusion_that_removes_nothing(self):
        """Verify an exclusion outside the selection fails rather than doing nothing.

        The label resolves to a real joint, so the usual "matches nothing" guard
        does not fire; without this check the joint would stay controlled.
        """
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "removed no joint"):
            select_joints(model, joints=["shoulder"], exclude_joints=["wrist"])

    def test_exclusion_is_scoped_to_the_selected_articulations(self):
        """Verify an exclusion naming a joint outside the selection is reported as unmatched.

        Resolving it model-wide instead would still fail, but as "removed no
        joint" rather than naming it as out of scope — hiding that the label was
        never selectable to begin with.
        """
        model = _fleet((_arm(), 2), (_leg(), 2))
        with self.assertRaisesRegex(ValueError, "none are regular tree joints of the selected"):
            select_joints(model, articulations=[0, 1], exclude_joints=["hip"])

    def test_rejects_exclusion_that_removes_everything(self):
        """Verify excluding the entire selection is rejected."""
        model = _fleet((_arm(), 2))
        with self.assertRaisesRegex(ValueError, "removed every selected joint"):
            select_joints(model, joints=["shoulder"], exclude_joints=["shoulder"])


if __name__ == "__main__":
    unittest.main()
