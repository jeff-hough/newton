Reject articulations mounted on a body they do not drive in `ControllerJointImpedance`.
`newton.eval_inverse_dynamics_passive` allocates per-body scratch with `wp.empty` and
relies on every body being written by an articulation traversal, so an articulation
attached to an external body read uninitialised memory and produced gravity and Coriolis
terms that varied between runs and between steps. This is now a construction error naming
the offending joints. The check applies only when a dynamics term is enabled, since a pure
PD configuration never evaluates inverse dynamics.
