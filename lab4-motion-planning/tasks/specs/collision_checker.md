# Spec: `CollisionChecker`

Implementation spec for `src/collision_checker.py`. It is written so that an
engineer (or an AI) can implement the module from this file alone, without
reading the existing implementation. Human-facing explanation lives in
`docs/01_collision_checking.md`; this file is the contract.

Words in **MUST / MUST NOT** are hard requirements. Anything not specified is
an implementation choice.

---

## 1. Goal

Answer three questions for the 6-DoF UR5e + mounted Robotiq 2F-85 gripper,
using **exactly the geometry MuJoCo executes** (not a separate collision URDF):

1. Is configuration `q` collision-free?
2. Is the straight joint-space segment `q1 → q2` collision-free?
3. What is the minimum signed distance between monitored geometry at `q`?

Consumers: `rrt_planner.py` (RRT / RRT*), `trajectory_smoother.py`
(shortcutting), `capstone_demo.py` and the recording scripts (clearance plots).

### Non-goals

- No collision checking through Pinocchio / hpp-fcl. Pinocchio is loaded only
  so that callers can use `cc.model` / `cc.data` / `cc.ee_fid` for FK.
- No continuous (swept-volume) collision checking; segments are sampled.
- No thread safety. One instance owns one `MjData`.
- No contact with the floor plane is ever reported.

---

## 2. Dependencies

From `lab4_common` (same `src/` directory, already exists — do not modify):

| Symbol | Meaning |
|---|---|
| `OBSTACLES` | `tuple[ObstacleSpec, ...]`, the 4 default boxes `box_1..box_4` |
| `ObstacleSpec` | frozen dataclass: `name`, `position`, `half_extents`, `rgba` |
| `URDF_PATH` | default UR5e URDF for Pinocchio |
| `load_mujoco_model(obstacle_specs=..., include_table=...)` | returns `(mj_model, mj_data)` for the canonical scene. Each obstacle becomes a **body named `spec.name`** with one box geom. With `include_table=True` the scene contains a **body named `"table"`** (top + legs, several geoms). |
| `load_pinocchio_model(urdf_path)` | returns `(model, data, ee_fid)` |

External: `mujoco`, `numpy`, `pathlib.Path`.

Scene facts the implementation relies on:

- The first 6 entries of `qpos` are the arm joints. `mj_model.nq` is larger
  than 6 (gripper joints follow).
- The floor plane belongs to the world body (body id 0).
- Robot body names: arm bodies are `shoulder_link`, `upper_arm_link`,
  `forearm_link`, `wrist_1_link`, `wrist_2_link`, `wrist_3_link`; every gripper
  body name starts with the prefix **`2f85_`**.
- Visual-only geoms have `geom_contype == 0`.

---

## 3. Public API

Module docstring + class docstring + type hints are required (project
standard). Use `from __future__ import annotations`.

```python
class CollisionChecker:
    def __init__(
        self,
        urdf_path: Path | None = None,
        obstacle_specs: list[ObstacleSpec] | tuple[ObstacleSpec, ...] | None = None,
        self_collision: bool = True,
        adjacency_gap: int = 1,
        include_table: bool | None = None,
    ) -> None: ...

    def is_collision_free(self, q: np.ndarray) -> bool: ...
    def is_path_free(self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.05) -> bool: ...

    def compute_min_distance(
        self,
        q: np.ndarray,
        include_self: bool = True,
        include_table: bool = True,
        include_obstacles: bool = True,
    ) -> float: ...
    def compute_min_environment_distance(self, q: np.ndarray, include_table: bool = True) -> float: ...
    def compute_min_obstacle_distance(self, q: np.ndarray) -> float: ...

    @property
    def num_collision_pairs(self) -> int: ...
```

Public / externally used attributes (other modules and tests read these —
names **MUST** match):

| Attribute | Type | Content |
|---|---|---|
| `obstacle_specs` | `list[ObstacleSpec]` | the resolved obstacle list |
| `self_collision`, `adjacency_gap` | as passed | |
| `mj_model`, `mj_data` | MuJoCo | from `load_mujoco_model` |
| `model`, `data`, `ee_fid` | Pinocchio | from `load_pinocchio_model(urdf_path or URDF_PATH)` |
| `_n_robot_geoms` | `int` | number of robot collision geoms |
| `_obstacle_ids` | `list[int]` | **all** environment geom ids (obstacle boxes **and** table geoms) |

Return types: `is_*` return Python `bool`; distance functions return Python
`float`.

---

## 4. Definitions

### 4.1 Resolving constructor arguments

```
if obstacle_specs is None:
    obstacle_specs = list(OBSTACLES)
    include_table  = True                     if include_table is None else include_table
else:
    obstacle_specs = list(obstacle_specs)
    include_table  = len(obstacle_specs) > 0  if include_table is None else include_table
```

So `CollisionChecker(obstacle_specs=[])` means **no environment at all**
(no boxes, no table) → self-collision only.

Then `mj_model, mj_data = load_mujoco_model(obstacle_specs=tuple(obstacle_specs), include_table=include_table)`.

### 4.2 Environment vs robot

- **Environment body names** = `{spec.name for spec in obstacle_specs}`,
  plus `"table"` if `include_table`.
- **Environment bodies** = bodies whose name is in that set.
- **Robot bodies** = every body that is not body 0, not an environment body,
  and not named `"world"`.
- **Robot geoms** = geoms with `geom_contype != 0` whose body is a robot body,
  in increasing geom-id order.
- **Environment geoms** = geoms with `geom_contype != 0` whose body is an
  environment body, in increasing geom-id order.
- **Table geoms** = environment geoms whose body is named `"table"`.
  **Obstacle geoms** = environment geoms that are not table geoms.

Because the floor is on body 0 it is in neither set, so floor contacts are
never reported.

### 4.3 Pairs

A pair is always stored normalized as `(min(g1, g2), max(g1, g2))`.

Monitored pairs are built in this order (keep a set for O(1) membership and
an insertion-ordered list without duplicates):

1. For every robot geom `r`, for every environment geom `e`: pair `(r, e)`.
   Put it into the **table bucket** if `e` is a table geom, else into the
   **obstacle bucket**.
2. If `self_collision`: for every unordered robot-geom pair `(a, b)`, `a`
   before `b` in robot-geom order, that passes the **self-pair rule** (4.4):
   put it into the **self bucket**.

`num_collision_pairs` = number of distinct monitored pairs.

### 4.4 Self-pair rule

A robot-robot geom pair `(a, b)` is monitored **only if all** hold:

1. `body(a) != body(b)`.
2. **NOT** (both body names start with `"2f85_"`). Internal gripper linkage
   pairs MUST be excluded (see Pitfall P1).
3. `tree_distance(body(a), body(b)) > adjacency_gap`.

`tree_distance(b1, b2)` = number of edges on the path between the two bodies
in the MuJoCo kinematic tree (`mj_model.body_parentid`, root = body 0):
walk `b1` up to the root recording the depth of each ancestor; walk `b2` up
until the first ancestor of `b1` is hit; distance = depth-in-b1-walk +
steps-from-b2. Parent–child = 1, siblings = 2.

Arm-vs-gripper pairs (e.g. `forearm_link` vs `2f85_base`) **are** monitored.

---

## 5. Algorithms

### 5.1 Setting a configuration (shared helper)

For every query at configuration `q` (length 6):

```
mj_data.qpos[:nq] = 0
mj_data.qvel[:nv] = 0
mj_data.qpos[:len(q)] = q
mujoco.mj_forward(mj_model, mj_data)
```

The gripper joints are therefore always reset to 0 (open). This MUST happen
on every query; the result MUST NOT depend on previous queries.

### 5.2 `is_collision_free(q)`

Set configuration, then iterate `mj_data.contact[0 : mj_data.ncon]`. Normalize
each contact's `(geom1, geom2)`. Return `False` on the first contact whose
pair is a monitored pair; otherwise `True`.

It uses MuJoCo's contact list (from `mj_forward`), **not** `mj_geomDistance`.

### 5.3 `is_path_free(q1, q2, resolution=0.05)`

```
diff = q2 - q1
dist = ||diff||_2
if dist < 1e-9: return is_collision_free(q1)
n = max(2, ceil(dist / resolution) + 1)
for i in 0 .. n-1:
    q = q1 + (i / (n - 1)) * diff
    if not is_collision_free(q): return False
return True
```

Both endpoints are checked. `resolution` is in radians (joint-space L2).

### 5.4 Minimum distance

`compute_min_distance(q, include_self, include_table, include_obstacles)`:

1. Collect pairs: obstacle bucket if `include_obstacles`, table bucket if
   `include_table`, self bucket if `include_self` (empty buckets contribute
   nothing).
2. Set configuration (always, even if no pairs).
3. If no pairs: return `float("inf")`.
4. For each pair call
   `mujoco.mj_geomDistance(mj_model, mj_data, g1, g2, 10.0, fromto)`
   with `fromto = np.zeros(6)`; take the minimum.
5. If `abs(min) < 1e-8` return `0.0`, else return `min` as `float`.

Sign convention: positive = clearance in metres, negative = penetration,
`0.0` = touching (MuJoCo returns 0 for touching geoms, including mechanically
adjacent arm/gripper geoms at `Q_HOME`).

Convenience wrappers:

- `compute_min_environment_distance(q, include_table=True)` =
  `compute_min_distance(q, include_self=False, include_table=include_table, include_obstacles=True)`
- `compute_min_obstacle_distance(q)` =
  `compute_min_distance(q, include_self=False, include_table=False, include_obstacles=True)`

### 5.5 Known semantics (intentional, do not "fix")

- `is_collision_free` sees only contacts MuJoCo itself generates, i.e. after
  MuJoCo's `contype`/`conaffinity` filter and its built-in parent–child
  exclusion. `mj_geomDistance` ignores those filters. A monitored pair can
  therefore have a negative distance yet never count as a collision. This is
  accepted: contacts are the boolean truth, distance is a clearance metric.
- All geom margins are 0 in this scene, so a contact means touching or
  penetrating. Should margins ever be non-zero, a contact within the margin
  still counts as a collision.
- "Visual" means `geom_contype == 0`; `conaffinity` is not used for
  classification (in this scene every robot collision geom is `(1, 1)`).
- Private bucket attribute names other than `_n_robot_geoms` and
  `_obstacle_ids` are free to choose; nothing outside the class reads them.
- The input `q` is not validated (no length or type checks).

---

## 6. Pitfalls (from `tasks/LESSONS.md`) — hard constraints

- **P1 — Gripper internals are not self-collision.** Robotiq finger links
  touch each other by design. If gripper-internal pairs are monitored,
  `Q_HOME` reports a negative distance / collision. → Rule 4.4.2.
- **P2 — One geometry truth.** Collision truth MUST come from the MuJoCo
  scene returned by `load_mujoco_model`, never from a separately loaded
  collision URDF/XML.
- **P3 — Do not disturb other users' state.** The checker owns its own
  `mj_data`; it MUST NOT accept or mutate an external `MjData`.
- **P4 — Visual geoms.** Geoms with `contype == 0` are visual meshes and MUST
  be ignored; including them inflates pair counts and produces false
  distances.

---

## 7. Acceptance criteria

Run from `lab4-motion-planning/`:
`pytest tests/test_collision.py` — all tests MUST pass.

In addition, these reference values were measured on the canonical scene and
MUST be reproduced exactly (counts) or with `atol=1e-6` (distances):

| Checker | `_n_robot_geoms` | env geoms (`len(_obstacle_ids)`) | table geoms | self pairs | obstacle pairs | table pairs | `num_collision_pairs` |
|---|---|---|---|---|---|---|---|
| `CollisionChecker()` | 23 | 9 | 5 | 146 | 92 | 115 | 353 |
| `CollisionChecker(obstacle_specs=[])` | 23 | 0 | 0 | 146 | 0 | 0 | 146 |
| `CollisionChecker(self_collision=False)` | 23 | 9 | 5 | 0 | 92 | 115 | 207 |

With `Q_HOME = [-π/2, -π/2, π/2, -π/2, -π/2, 0]` and
`Q_COLL = [0.84, -2.0, 2.0, 2.13, -0.79, 4.18]`:

| Call | Expected |
|---|---|
| `cc.is_collision_free(Q_HOME)` | `True` |
| `cc.compute_min_distance(Q_HOME)` | `0.0` |
| `cc.compute_min_environment_distance(Q_HOME)` | `0.121394597` |
| `cc.compute_min_obstacle_distance(Q_HOME)` | `0.327200207` |
| `cc.is_collision_free(Q_COLL)` | `False` (forearm through `box_2`) |
| `cc.compute_min_distance(Q_COLL)` | `-0.049227345` |
| `cc_no_env.is_collision_free(np.zeros(6))` | `True` |
| `cc_no_env.is_collision_free([0, -0.5, 2.8, 2.0, 0, 0])` | `True` |
| `cc_no_env.compute_min_distance(Q_HOME)` | `0.0` |
| `cc.is_path_free(Q_HOME, Q_COLL, resolution=0.1)` | `False` |

Where `cc = CollisionChecker()` and `cc_no_env = CollisionChecker(obstacle_specs=[])`.
