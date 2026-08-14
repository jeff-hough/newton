Add `newton.controllers.ControllerDifferentialKinematics` and
`ControllerDifferentialKinematicsModelFree`, one-step differential inverse kinematics driving a
labelled site toward a target pose. The model-based variant takes a finalized `newton.Model` and
selects what to control with `articulations` and `joints`, matching `ControllerJointImpedance`;
both expose strictly typed `Inputs` and `Outputs`. Five Jacobian-inverse methods are available via
`IkMethod`, with optional orientation weighting and null-space joint-limit avoidance whose limits
default to the model's own. Heterogeneous fleets are supported: only the Jacobian is padded, and a
short robot's padding columns are ignored by the solver. `site` accepts several labels, so one
controller can span robot kinds whose end-effector sites are named differently. The
`controller_diff_ik_heterogeneous` example drives a mixed fleet of two Franka FR3 arms and two
UR10s, with the Frankas' gripper fingers excluded from control.
