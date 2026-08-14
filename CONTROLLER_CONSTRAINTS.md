# Newton controllers: goals and constraints

What any implementation of these controllers has to achieve, and what Newton
forces on it. Written to outlive the current code — if the implementation is
thrown away and restarted, this is the part worth keeping.

Everything under "Constraints" was measured against Newton, not recalled.

---

## 1. Goals, in priority order

All three matter. The order is which one to bend the design around when they
conflict.

### Goal 1 — add a controller to a world you already built

The overwhelmingly common case, and the one that must be **simple**:

```python
model = scene.finalize()          # arms, conveyor, doors, furniture, the lot

ctrl = ControllerJointImpedance(
    model,
    articulations=["arm"],        # or joints=[...]; say what you control
    stiffness=kp, damping=kd,
)
```

Acceptance criteria:

- **No index arrays.** Not optional-but-encouraged — none at all.
- The controller's model *is* the simulation's model, so binding is direct:
  `inputs.joint_q = state.joint_q`, no copy.
- Uncontrolled parts of the scene are read (so forward kinematics is right) and
  never driven.
- Selecting what to control costs work proportional to *joints per robot*, not
  to the number of robots.

### Goal 2 — bring your own model, and wire it to the simulation

For deliberate modelling error, or a controller model that is only part of the
scene. Must be **achievable and simple**, though necessarily less simple than
goal 1:

```python
ctrl = ControllerJointImpedance(
    arm_model,                    # 3 scalar joints, masses deliberately wrong
    model_coord_to_sim_coord=...,  # where my coordinates live in your arrays
    model_dof_to_sim_dof=...,      # where my DOFs live
    stiffness=kp, damping=kd,
)
```

Acceptance criteria:

- When the two models are built the same way (same sub-builder, same order),
  the mapping is the identity and **nothing extra is passed**. Goal 2 degrades
  to goal 1 for the common case.
- When they genuinely differ, the user states the correspondence **once**, and
  everything else is derived from it.
- The user never computes a composed mapping by hand. Given "where my model
  lives in your simulation" plus "what I control", the controller derives every
  gather and scatter index itself.

### Goal 3 — targets that arrive in simulation layout

Joint targets may come from another process or another controller, laid out in
simulation order rather than one-entry-per-controlled-DOF. The user must be able
to **override the compact input**.

Acceptance criteria:

- The override is expressed **by name**, in the same idiom as the rest of the
  selection API — mapping joint target names to DOF indices, the way the other
  ports get their mapping. Not raw index arrays, and not a mode flag the user
  has to remember to set.
- The default stays compact: one entry per controlled DOF, no override needed.
- **It must remain checkable.** This is the hard part. Today `joint_q_des` must
  be exactly `controlled_dof_count` long, and that single rule catches a whole
  class of wiring mistakes at construction. Any override mechanism has to keep a
  wrongly sized or wrongly wired array detectable — if the answer is "any length is
  legal depending on configuration", the check has been thrown away.
- Chaining two Newton controllers (differential IK produces joint targets,
  impedance tracks them) must not require the user to write a gather kernel.

---

## 2. Constraints from Newton

Measured. Each of these has cost real debugging time at least once.

### Positions and velocities are indexed differently

A free joint spans **7 coordinates but 6 DOFs**, so everything behind it is
offset differently in the two spaces:

```
joint_type    : [FREE, REVOLUTE, REVOLUTE, REVOLUTE]
joint_q_start : [0,  7,  8,  9, 10]     <- State.joint_q            (coordinates)
joint_qd_start: [0,  6,  7,  8,  9]     <- State.joint_qd, joint_f  (DOFs)
```

Any controller reading both needs **two** mappings. One array cannot serve both,
and conflating them reads a quaternion component as a joint angle. This is not a
design choice — it is the data model.

`Control.joint_target_q` adds a wrinkle: it is coordinate-shaped or DOF-shaped
depending on `newton.use_coord_layout_targets`, snapshotted per `Model`.
`joint_target_qd` is always DOF-shaped.

### `ModelBuilder.finalize()` mutates the builder

Ten fields change on a one-joint builder, including `world_count: 0 -> 1`:

```
articulation_world_start, body_world_start, default_joint_cfg,
joint_constraint_world_start, joint_coord_world_start, joint_dof_world_start,
joint_world_start, particle_world_start, shape_world_start, world_count
```

A controller that takes a `ModelBuilder` and finalizes it therefore has no clean
ownership story, and building a second controller from the same builder is a
different operation from building the first. Take a finalized `Model`.

### `eval_fk` runs per articulation

A joint outside every articulation **never updates its body transform**.
Verified by poisoning `body_q` and observing that forward kinematics leaves it
poisoned while computing children relative to it. An uncontrolled joint must be
*inside* the articulation to be read; putting it outside gives a stale base.

### `eval_inverse_dynamics_passive` is nondeterministic for externally mounted articulations

`_InverseDynamicsScratchBuffer` allocates per-body scratch with `wp.empty`
(`inverse_dynamics.py:57-67`) on the invariant stated at `:463` — *"fully
overwrites every body's scratch"* — which holds only for bodies reached by the
traversal being run. An articulation attached to a body no articulation drives
reads recycled memory.

Measured on identical models with provably identical, constant `body_q`:

| model | gravity, two consecutive calls | stable |
|---|---|---|
| arm alone | `-0.7063`, `-0.7063` | yes |
| arm + unrelated non-articulated body | `-0.7063`, `-0.7063` | yes |
| arm on a **fixed** mount outside the articulation | `-0.7782`, `-0.9507` | **no** |
| arm on a **revolute** mount outside the articulation | `-0.7782`, `-0.9507` | **no** |
| arm mounted on another *articulation's* body | unstable with `mask=None` too | **no** |

Not DOF-dependent (a zero-DOF mount is corrupted just as badly), not mask-related,
and not confined to the articulation's root joint — an articulation may be a
forest whose first joint is world-rooted while a later one hangs off an external
body.

**This is a Newton defect, not a controller one, and it should be reported
upstream regardless of what happens to this code.** The fix is `wp.zeros` for the
spatial scratch, or a traversal covering ancestor bodies. Until then a controller
must reject such models, because nothing downstream can detect the corruption.

### Loop-closing joints sit outside `articulation_end`

`articulation_start` bounds them; `articulation_end` does not:

```
articulation_start: [0 4 5]     articulation_end: [3 5]     joint 3 = 'loop'
```

They are ordinary revolute joints, so no joint-type check excludes them.
Membership must be tested against `articulation_end`.

### `replicate()` copies labels verbatim

Not suffixed. A thousand replicas are all literally labelled `"arm"`, which is
what lets one string address a fleet. Consequently, default labels **collide**:
`ModelBuilder` numbers unlabelled joints per sub-builder, so two unrelated robot
kinds both contain `joint_1` and both contain `articulation_0`.

### Zeroing a joint's gains does not free it

`Kp = Kd = 0` still yields gravity-compensation torque — `-0.4905 N·m` measured,
versus exactly `0.0` when the joint is excluded from the selection. The PD term
vanishes; `τ = M·(…) + C + g` does not.

### Padded operator columns being zero makes ragged fleets nearly free

`J Jᵀ` and `Jᵀy` sum over a padded column that contributes nothing, so a solver
needs no per-robot bounds. Only code reading *per-DOF* data alongside it (joint
limits, for instance) needs the real count.

---

## 3. Conclusions that held up

- **Select by joint and label, not by index arithmetic.** Integers match
  exactly; labels match everything carrying them. No globbing, no heuristics.
- **Flat joint-space vectors.** Pad only operators (mass matrix, Jacobian) —
  they are not joint-space vectors and nobody expects them to be 1-D.
- **Exact versus minimum shape is decided by ownership.** Arrays the controller
  sizes are checked exactly; arrays the caller owns and the controller indexes
  into are checked as a lower bound. Ownership is the rule, not the port name.
- **One derivation seam, testable without physics.** An index bug should surface
  as a wrong integer in a sub-second test, not as a wrong float three layers
  downstream. That is how the original DOF-mapping bug hid.
- **Structural checks over heuristics.** A shared gain array whose length must
  equal the controlled DOF count catches over-matching precisely. A heuristic
  that guesses whether two robots are "the same kind" has false positives and
  blocks legitimate use.

---

## 4. Things that cost time

- Adding configuration before a case existed for it. Every knob doubles a code
  path and its test surface, and creates precedent pressure for the next one.
- Porting the previous API forward incrementally instead of deriving from the
  constraints above. Padded gains, per-port index overrides, and configurable
  port names all came from carrying the old shape along.
- Tests that pin nothing. Mutation testing — deliberately break the code, check
  a named test fails — repeatedly found assertions that passed against a broken
  implementation. Convergence tests in particular cannot see a wrong Jacobian,
  because a wrong Jacobian is often still a descent direction.
