Rework `ControllerJointImpedance` around joint selection. The controller now takes a
finalized `newton.Model` and selects what to control with `articulations` and `joints`,
given as model indices or labels; labels match every joint carrying them, so addressing a
replicated fleet costs one string per joint rather than one integer per DOF. Gains and
targets are flat arrays of length `controlled_dof_count`, ordered by articulation then by
model joint index, replacing the padded `(robot_count, max_dofs)` layout. The six per-port
index overrides are replaced by two optional mappings, `model_coord_to_sim_coord` and
`model_dof_to_sim_dof`, needed only when the controller's model is not the simulation's.
Only the *controlled* joints must be 1-DOF revolute or prismatic; free and ball joints
elsewhere in the model are read for forward kinematics and left unactuated.
