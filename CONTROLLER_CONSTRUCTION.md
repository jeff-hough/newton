# Controller construction: what changed and why

Notes on the reworked constructor shared by `ControllerJointImpedance` and
`ControllerDifferentialKinematics`. Not official documentation — this exists to
explain the diff.

---

## The change in one view

**Before**

```python
ControllerJointImpedance(
    builder,                     # newton.ModelBuilder
    *,
    default_dof_indices,         # wp.array[uint32], controller slot -> sim slot
    stiffness, damping,          # (robot_count, max_dofs), padded
    joint_q_idx=None,            # six per-port index overrides
    joint_qd_idx=None,
    joint_q_des_idx=None,
    joint_qd_des_idx=None,
    joint_qdd_idx=None,
    joint_f_idx=None,
    device=None,
    ...
)
```

**After**

```python
ControllerJointImpedance(
    model,                       # newton.Model, already finalized
    *,
    articulations=None,          # which robots   (indices or labels)
    joints=None,                 # which joints   (indices or labels)
    model_coord_to_sim_coord=None,   # only if the controller's model isn't the sim's
    model_dof_to_sim_dof=None,
    stiffness, damping,          # (controlled_dof_count,), flat
    ...
)
```

Seven index arrays became two, and both default to `None`. The differential IK
controller takes the same arguments plus `site=`.

The four things this buys are below.

---

## 1. The builder ownership contract was unanswerable

The old constructor took a `ModelBuilder` and called `finalize()` on it
internally. That forces a question with no good answer: **does the controller own
the builder you handed it?**

It cannot simply borrow it, because `finalize()` mutates the builder. Measured on
a one-joint builder, ten fields change:

```
articulation_world_start, body_world_start, default_joint_cfg,
joint_constraint_world_start, joint_coord_world_start, joint_dof_world_start,
joint_world_start, particle_world_start, shape_world_start, world_count

world_count: 0 -> 1
```

So constructing a controller silently altered the caller's builder, and building
a second controller from the same builder was a different operation from building
the first. The options were all bad: document "we mutate your builder," deep-copy
it (there is no clear copy contract for a `ModelBuilder` — it can share meshes and
other objects), or keep quiet.

**Taking a finalized `Model` removes the question.** The model is borrowed and
never written to, so the contract is one line, and you finalize once and can hand
the same model to as many controllers as you like.

```python
model = scene.finalize()          # you own this, you did it
impedance = ControllerJointImpedance(model, ...)
diff_ik   = ControllerDifferentialKinematics(model, site="ee", ...)
```

---

## 2. It opens the case people actually want: control part of a real scene

This is the important one, and it was not previously expressible.

The old design assumed the model you gave the controller *was* the thing being
controlled — every DOF in it was controlled, and `default_dof_indices` told the
controller where those DOFs lived in your simulation arrays. A controller built
from your whole scene would try to drive the whole scene.

Now you pass the scene you already have and say what to control:

```python
model = scene.finalize()          # arms, conveyor, doors, furniture, the lot

ctrl = ControllerJointImpedance(
    model,
    articulations=["arm"],        # every robot labelled "arm"
    stiffness=kp, damping=kd,
)
```

No index arrays. The mapping to your simulation arrays is the identity, because
the controller's model *is* your simulation's model, so `inputs.joint_q` binds
straight to `state.joint_q` with no copy.

Two properties make this work, and both are new:

- **The read set and the write set are different sizes.** Every model coordinate
  and DOF is read each step, so uncontrolled joints still contribute to forward
  kinematics — an arm bolted to a door gets the right base transform. Only the
  selected joints receive commands.
- **Only *controlled* joints must be 1-DOF.** A floating base, a ball joint, a
  6-DOF gantry elsewhere in the scene is read and left alone. The old code
  rejected any model containing one, which ruled out most realistic scenes.

---

## 3. A fully independent model still works

Nothing about the above prevents giving the controller its own model — which is
the point when you *want* modelling error between the controller and the plant.

The common case needs no extra arguments. Build the controller's model the same
way you built the simulation's, and the DOF orders line up:

```python
scene = newton.ModelBuilder(); scene.replicate(arm, world_count=1000)
twin  = newton.ModelBuilder(); twin.replicate(arm_with_wrong_masses, world_count=1000)

ctrl = ControllerJointImpedance(twin.finalize(), stiffness=kp, damping=kd)
```

When the two genuinely disagree — the controller holds only the arm while the
simulation also has a floating base and furniture — you say where its DOFs live:

```python
ctrl = ControllerJointImpedance(
    arm_model,
    model_coord_to_sim_coord=wp.array([7, 8, 9], dtype=wp.int32),
    model_dof_to_sim_dof=wp.array([6, 7, 8], dtype=wp.int32),
    stiffness=kp, damping=kd,
)
```

### Why that is two arrays and not one

Newton indexes positions and velocities differently. A free joint spans **7
coordinates but 6 DOFs**, so everything behind it is offset differently in the
two spaces:

```
joint_type    : [FREE, REVOLUTE, REVOLUTE, REVOLUTE]
joint_q_start : [0,  7,  8,  9, 10]     <- State.joint_q          (coordinates)
joint_qd_start: [0,  6,  7,  8,  9]     <- State.joint_qd, joint_f (DOFs)
```

The three arm joints are at coordinates 7,8,9 and DOFs 6,7,8 — the same joints,
different numbers. A single mapping array cannot serve both, and conflating them
reads a quaternion component as a joint angle. The old code sidestepped this by
rejecting any model containing a multi-DOF joint; that restriction is gone, so
the split is explicit instead.

Both arrays default to the identity, so you only meet them in the case that
needs them.

---

## 4. Selection replaces index arithmetic

`articulations` and `joints` both take **model indices or labels**:

```python
ControllerJointImpedance(model, ...)                                # everything
ControllerJointImpedance(model, articulations=[0, 5, 17], ...)      # three instances
ControllerJointImpedance(model, articulations=["arm"], ...)         # every "arm"
ControllerJointImpedance(model, joints=["shoulder", "elbow"], ...)  # those joints, fleet-wide
```

**Integers match exactly; labels match every entry carrying them** — no
exceptions, in either direction. If two robot kinds share a joint label,
including it takes both and excluding it drops both. That second rule is what
makes fleets cheap: `replicate()` copies labels verbatim, so a
thousand robots are all literally labelled `"arm"` and one string addresses them
all. Selection cost scales with joints *per robot*, not with robot count.

Matching is exact string equality — there is no `arm*` globbing, and `"arm"` does
not match `"arm1"`.

There is deliberately no cleverness beyond that. An earlier version rejected a
label whose matching articulations had differing joint layouts, on the theory
that it was probably a mistake. It was a heuristic on label *signatures*, not a
fact about robot kinds, and it had false positives — two copies of the same arm,
one bolted down with an extra fixed joint, were "ambiguous" — while blocking the
legitimate case of dropping every `gripper` across a mixed fleet. The failure it
guarded against is already caught, loudly, by the gain length.

### Set algebra, when include-only isn't enough

Two shapes are awkward to express by inclusion alone: a robot that is only
partly controlled, and a fleet with one kind left out. `select_joints()` handles
both and returns indices for `joints=`:

```python
from newton.controllers import select_joints

# an arm with a free gripper — one string, however many replicas there are
controlled = select_joints(model, exclude_joints=["gripper"])

# ten robot kinds, control nine
controlled = select_joints(model, exclude_articulations=["crane"])

ctrl = ControllerJointImpedance(model, joints=controlled, ...)
```

### Guardrails

Selection is the easiest place to be silently wrong, so it fails loudly instead:

- A selector matching nothing is an error, never an empty selection.
- Duplicates are rejected — two DOFs writing one output slot would race.
- An exclusion that removes nothing, or everything, is an error.
- Selecting more than you meant is caught by the **gain array's length**, which
  must equal `controlled_dof_count` exactly. If a label matched four joints and
  you sized the gains for two, construction fails with
  `stiffness must have shape (4), got (2,)` — which names the real count.

### Ordering

Controlled DOFs are ordered **by articulation, then by model joint index** — not
by the order you listed them. `joints=["wrist", "shoulder"]` gives the same
layout as `joints=["shoulder", "wrist"]`.

That is what makes fleet gains a one-liner, since the layout is robot-major:

```python
kp = np.tile([400.0, 300.0, 150.0], 1000)
```

---

## Two things that follow from the rework

**Gains and targets are flat.** They were `(robot_count, max_dofs)` with padding;
they are now `(controlled_dof_count,)`. Heterogeneous fleets stop costing
anything — robots of 3 and 1 DOFs give a flat array of 4, with no padding entries
to fill in:

```python
kp = np.array([200., 200., 200., 200.])     # not [[200,200,200],[200,0,0]]
```

The mass matrix and the Jacobian are still blocked and padded per robot, because
they are operators rather than joint-space vectors.

**"Robot" left the model-based API.** It only ever meant "selected articulation."
It survives in the model-free classes, which have no model and no articulations,
so a robot there really is just a group of DOFs you declare.

---

## Migrating

```python
# before
ctrl = ControllerJointImpedance(
    builder,
    default_dof_indices=wp.array(np.arange(n_dofs), dtype=wp.uint32),
    stiffness=wp.array(kp_2d), damping=wp.array(kd_2d),   # (robots, max_dofs)
)

# after
model = builder.finalize()                                # you call this now
ctrl = ControllerJointImpedance(
    model,
    stiffness=wp.array(kp_flat), damping=wp.array(kd_flat),   # (controlled_dof_count,)
)
```

If your simulation arrays followed the model's own layout — the overwhelmingly
common case, and what `default_dof_indices=arange(...)` meant — you pass no index
arrays at all.

Ports are typed rather than duck-typed namespaces: `input()` and `output()`
return `Inputs`/`Outputs` objects with fixed field names, and fields for disabled
features are `None`.

---

## Working on `_dof_indices.py`

All of the index derivation lives in one file:

```
newton/_src/controllers/dof_indices.py
```

It sits in the shared layer next to `kernels.py` and `utils.py`, because both
controllers depend on all of it — they read the same eight `_DofIndices` fields,
not overlapping subsets.

### Why it is a separate file

Index derivation is where this code has actually been wrong. The bug that
started the rework was a DOF-mapping error that only surfaced as *wrong torque
values*, three layers downstream of where it happened. Keeping the derivation in
one pure, host-side function means it can be tested directly:

```python
# a floating base, then three revolute joints; control the 1st and 3rd of them
idx = _build_dof_indices(model, joints=[1, 3])

assert_np_equal(idx.controlled_dof_to_model_dof.numpy(), [6, 8])    # DOF space
assert_np_equal(idx.controlled_dof_to_model_coord.numpy(), [7, 9])  # coordinate space
```

No controller, no solver, no gains, no physics step. A mistake shows up as a
wrong integer instead of a wrong float — and this particular example is one the
old code got wrong, because it conflated the two spaces.

### The flow

```
user selection  ->  _resolve_articulations  ->  which articulations
                    _resolve_joints         ->  which joints, within those
                    _build_dof_indices      ->  _DofIndices (device arrays)
                                            ->  kernels do arithmetic on it
```

Everything in `_DofIndices` is recomputable from `(model, selection)`. It holds
no user-supplied mapping — `model_coord_to_sim_coord` and `model_dof_to_sim_dof`
stay on the controller, because they are not derived from the model.

### `select_joints(...)` — the only public entry point

```python
select_joints(model, *, articulations=None, exclude_articulations=None,
                        joints=None, exclude_joints=None) -> list[int]
```

Set algebra over the selection, returning plain model joint indices to hand to a
controller's `joints=`. **Use it when** you need something inclusion alone cannot
express: a partly-controlled robot, or a fleet with one kind left out.

It resolves articulations first, so `exclude_joints` only ever sees the
articulations that survived `exclude_articulations`.

### `_build_dof_indices(model, *, articulations, joints) -> _DofIndices`

What the controllers call. Resolves the selection, rejects non-scalar controlled
joints, decides the ordering, and uploads the tables.

**Use it when** adding a third controller. Do not re-derive indices yourself —
that is how two encodings of the same fact drift apart.

It is the single place ordering is decided:

```python
order = np.lexsort((selected_joints, owning_art))   # by articulation, then joint index
```

Changing that changes the meaning of every per-DOF gain array, so it is load
bearing, not incidental.

### `_DofIndices` — the table

Five device arrays and four host scalars. What consumes each:

| field | shape | who reads it |
|---|---|---|
| `controlled_dof_to_model_dof` | `(n_dofs,)` | gravity/Coriolis readback; Jacobian column selection; composed output mapping |
| `controlled_dof_to_model_coord` | `(n_dofs,)` | reading `joint_q`, which is coordinate-indexed |
| `dof_offsets` | `(robots + 1,)` | CSR robot boundaries: padding guards, flat↔padded conversion, `dofs_per_robot` for the inner controller |
| `selected_articulations` | `(robots,)` | mass-matrix block index; Jacobian articulation index; the `eval_*` mask |
| `articulation_dof_start` | `(robots,)` | model DOF → articulation-local DOF, for indexing `H` and `J` |
| `controlled_joints` | tuple | debugging, and the controller's public property |
| `robot_count`, `controlled_dof_count`, `max_dofs` | int | buffer sizes and launch dimensions |

The two `controlled_dof_to_model_*` arrays exist separately because Newton
indexes positions and velocities differently — see §3. They are equal only when
nothing upstream spans more coordinates than DOFs.

`articulation_dof_start` is the one field that is a convenience rather than a
necessity: it could be recomputed in-kernel as
`joint_qd_start[articulation_start[a]]`, at the cost of two more model arrays in
every kernel signature.

### The three private resolvers

**`_resolve_articulations(model, articulations) -> np.ndarray`**
Indices or labels to sorted, unique articulation indices. `None` means all.
Rejects out-of-range indices, unknown labels, and duplicates.

**`_resolve_joints(model, joints, selected_articulations) -> np.ndarray`**
Indices or labels to joint indices, scoped to those articulations. `None` means
every scalar joint. Returns **resolution order, not sorted** — ordering is
`_build_dof_indices`'s job, deliberately, so there is one place to change it.

Its error messages distinguish three cases, which is worth preserving: no such
label anywhere; the label exists but names a joint outside the selection
(loop-closing joints look like this); and the label matched nothing in scope.

**`_joint_to_articulation(model, joints) -> np.ndarray`**
Which articulation owns each joint. The subtlety is that membership is bounded by
`articulation_end`, **not** `articulation_start[i + 1]`: the gap between them
holds loop-closing joints, which are ordinary revolute joints that no type check
would exclude.

### Invariants to keep

- **Everything derives from `(model, selection)`.** If something needs a
  user-supplied array, it belongs on the controller, not in the table.
- **One encoding per fact.** An earlier version stored both a 2-D `local_dofs`
  array and the flat model-DOF list — two encodings of one selection, built by
  separate code paths, free to drift. Prefer arithmetic on the flat array.
- **Ordering is decided once**, in `_build_dof_indices`.
- **A selector matching nothing is an error**, never an empty selection.
- **No cleverness in label matching.** Exact equality, take everything that
  matches. Selecting more than intended is caught by the gain array's length.

### Testing changes to it

`newton/tests/test_controllers_dof_indices.py` covers this file directly and
runs in about a second. Worth exercising when you touch it, because each has
caught a real defect:

- coordinate/DOF divergence behind a free joint
- non-contiguous joint subsets within one articulation
- loop-closing joints, which sit between `articulation_end` and the next
  `articulation_start`
- heterogeneous fleets, where `dof_offsets` is ragged
- articulations whose DOFs do not start at model DOF 0 — a single-articulation
  model leaves both the block index and the base offset at zero and cannot
  detect an error in either

Mutating the file and re-running is a cheap check that a test actually pins what
you think it does. Several here were vacuous until a deliberate defect was tried
against them.
