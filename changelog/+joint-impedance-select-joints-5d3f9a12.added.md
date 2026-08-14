Add `newton.controllers.select_joints()`, which resolves an articulation and joint selection to
model joint indices, with `exclude_articulations` and `exclude_joints` for the two shapes plain
selection cannot express: a robot that is only partly controlled, since `joints` applies within
every selected articulation, and a fleet with one robot kind left out, since listing the kinds you
keep scales with the fleet rather than with the exclusion. Exclusions take labels, so dropping
every gripper in a replicated fleet, or one kind from a ten-kind scene, costs one string. The
result is passed to the `joints` argument of `ControllerJointImpedance` or
`ControllerDifferentialKinematics`.
