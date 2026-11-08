# Lab 2 — Lessons Learned

Bugs found while setting up, running and reading Lab 2 (UR5e).
Format per bug: **Symptom / Root cause / Fix / Takeaway**.
Background theory (actuators, feedforward, dynamics, Pinocchio API) lives in
[CONTROL_DYNAMICS_NOTES.md](CONTROL_DYNAMICS_NOTES.md).

Environment: Ubuntu 22.04, Python 3.10.12, ROS 2 Humble sourced in `~/.bashrc`.
Venv: `.venv/` at repo root — mujoco 3.16.0, pin 4.1.0, numpy 2.2.6.

| # | Bug | Area | Status |
|---|---|---|---|
| 1 | Segfault on `import pinocchio` in the venv | Environment | Fixed (2026-10-16) |
| 2 | Missing Menagerie model / meshes | Environment | Fixed (2026-10-16) |
| 3 | 133 mm tracking error in capstone `c3` | Control | Fixed upstream |
| 4 | `b2` / `c1` send torques into position servos | Control | **Open** — confirmed locally |
| 5 | Stale lesson notes after the scene was rebuilt | Documentation | Open |
| 6 | `c3` draws 3 face diagonals instead of 3 cube edges | Trajectory | **Open** — confirmed locally |
| 7 | `b1` trapezoidal: synchronized joints miss the goal (~11°) | Trajectory | Fixed (2026-10-23) |

---

## Environment

### 1. Segfault on `import pinocchio` inside the venv

**Symptom**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 ...
  File "/opt/ros/humble/lib/python3.10/site-packages/pinocchio/__init__.py", line 19
AttributeError: _ARRAY_API not found
Segmentation fault (core dumped)
```
After unsetting only `PYTHONPATH`, a second error:
```
ImportError: .../.venv/.../pinocchio_pywrap_default...so:
  undefined symbol: EIGENPY_ARRAY_APIPyArray_RUNTIME_VERSION
```

**Root cause**
`source /opt/ros/humble/setup.bash` (in `~/.bashrc`) exports:
- `PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:...` → Python imports
  ROS's pinocchio (built against NumPy 1.x) instead of the venv's `pin` wheel.
- `LD_LIBRARY_PATH=/opt/ros/humble/lib:...` → even with the right Python package,
  the dynamic linker loads ROS's `libeigenpy.so`, which is ABI-incompatible with
  the venv's pinocchio.

`PYTHONPATH` entries take precedence over the venv's `site-packages`, so a venv
alone does not isolate from ROS.

**Fix**
Appended a block to `.venv/bin/activate` that saves and unsets `PYTHONPATH` and
`LD_LIBRARY_PATH` on activation, and restores them in `deactivate`.
Without activation, run with:
```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python ...
```

**Takeaway**
On a machine with ROS sourced, a pip venv is not isolated: check
`echo $PYTHONPATH $LD_LIBRARY_PATH` and `python -c "import pinocchio; print(pinocchio.__file__)"`
first. Never run `.venv/bin/python` directly — always `source .venv/bin/activate`.

---

### 2. Missing Menagerie model / meshes

**Symptom**
`models/mujoco_menagerie/` and the `models/assets` symlink did not exist, so
`lab_scene.xml` could not `<include>` the UR5e (and meshes would fail with
`Error opening file '.../base_0.obj'`). The `tools/setup_env.sh` referenced in
`lab2_navigation.md` is not in this repo.

**Root cause**
Menagerie is gitignored (fetched separately), and MuJoCo resolves the included
file's `meshdir="assets"` relative to the **top-level** scene file, so
`models/assets` must exist.

**Fix**
```bash
cd lab2_ur5e/models
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set universal_robots_ur5e robotiq_2f85  # Lab 3+ needs the gripper
ln -s mujoco_menagerie/universal_robots_ur5e/assets assets
```

**Takeaway**
After a fresh clone, these two steps are required before any test or demo
runs. A sparse checkout fetches only the UR5e folder instead of the whole
Menagerie repo.

---

## Control

Bugs 3 and 4 share one root cause: **the actuator definition was not read
before writing the controller.** See
[CONTROL_DYNAMICS_NOTES.md §1](CONTROL_DYNAMICS_NOTES.md#1-two-kinds-of-actuator).

### 3. 133 mm tracking error in capstone `c3` (fixed upstream)

**Symptom**
The end-effector drawing the cube deviated **133 mm RMS** from the trajectory.

**Root cause**
The original code (no longer in git) was `data.ctrl[:6] = tp.q`, i.e. the
desired angle was sent straight to the actuator. But the Menagerie actuators
are **position servos** (a PD spring), not "set the angle". The spring must
deflect to produce force, so:
1. gravity makes the arm sag,
2. damping resists the *desired* motion too, causing ~0.2 s lag,
3. the total torque exceeds ±150 N·m and saturates.

**Fix**
Feedforward hidden inside the servo setpoint
([CONTROL_DYNAMICS_NOTES.md §2](CONTROL_DYNAMICS_NOTES.md#2-how-the-feedforward-in-c3-works)):
```python
ctrl = q_des + qfrc_bias / Kp + Kd * qd_des / Kp
```
Result: **0.088 mm RMS**, ~1500× better.
Locations: `src/c3_draw_cube.py:294–295` (settle phase) and `:332–333` (main
loop); `src/c3_record_video.py:93` and `:119`.
Also logged upstream in `tasks/lessons.md` (Turkish) of
`ozkannceylan/mujoco-robotics-lab`.

**Takeaway**
With a position servo, `ctrl = q_des` always leaves an error: feedback only
produces force by being wrong. Supply the known force (gravity, damping) with
feedforward instead.

---

### 4. `b2` and `c1` send torques into position servos (open)

**Symptom**
Running `src/b2_control_hierarchy.py` locally (2026-10-16, Menagerie `62ab00f`):

| Controller | RMS joint err | RMS EE err | Mean \|τ\| | Max \|τ\| |
|---|---|---|---|---|
| PD+g | 0.051 rad | 15.8 mm | 185 N·m | 278 N·m |
| Computed torque | 0.676 rad | **63.8 mm** | 104 N·m | 188 N·m |
| Impedance | 0.878 rad | **160.4 mm** | 13 N·m | 28 N·m |

The more model-based the controller, the *worse* it tracks — the opposite of
what theory predicts. Mean |τ| of 185 N·m also exceeds the ±150 N·m limit.

**Root cause**
`b2_control_hierarchy.py:273–277` computes `tau = controller_fn(...)` and calls
`sim.set_ctrl(tau)` → `mujoco_sim.py:102`: `data.ctrl[:6] = ctrl`.
`c1_pick_and_place.py:226` does the same.
PD+g, computed torque, impedance and OSC all output **N·m**, but the scene uses
the same position-servo actuators as `c3`
(`<general gainprm="2000" biasprm="0 -2000 -400">`). MuJoCo reads 40 N·m as a
**target angle of 40 rad**, then clips it to `ctrlrange` ±2π.

Unexplained: upstream `docs/b2_controller_summary.csv` reports computed torque
at 3.6 mm, which contradicts the run above. It was probably generated with an
earlier (lost) scene that used torque motors.

**Fix (proposed, not applied)**
Turn the first 6 actuators into torque motors at runtime, in `b2` and `c1`
only — `c3` must keep the position servos:
```python
m = sim.model
m.actuator_gainprm[:6, :] = 0;  m.actuator_gainprm[:6, 0] = 1     # τ = 1·ctrl
m.actuator_biasprm[:6, :] = 0                                       # remove spring + damper
m.actuator_biastype[:6] = mujoco.mjtBias.mjBIAS_NONE
m.actuator_ctrlrange[:6] = m.actuator_forcerange[:6]                # ±150 / ±28 N·m
```
Then rerun `b2` and expect CT < PD+g in tracking error. A closed-loop test for
`b2` (extension 2 in `lab2_navigation.md`) would have caught this.

**Takeaway**
The same line `data.ctrl[:] = x` means an angle or a torque depending on the
model. Read the `<actuator>` block before writing any controller, and add a
closed-loop test, not only unit tests of the control law.

---

## Documentation

### 5. Stale lesson notes after the scene was rebuilt (open)

**Symptom**
Upstream `tasks/lessons.md` mentions a table at z ≈ 0.41 m and
`offwidth="1920"`, but `models/lab_scene.xml` (rebuilt 2026-08-13) has neither.

**Root cause**
When the scene was rebuilt, the existing lessons were not checked against it.

**Fix**
Pending: re-check each lesson against the current scene and mark the obsolete
ones.

**Takeaway**
A lesson that names a file, parameter or value can go stale. When a file
changes, grep the lessons for it.

---

## Trajectory

### 6. `c3` draws 3 face diagonals instead of 3 cube edges (open)

**Symptom**
In the viewer the red trail leaves the green wireframe on three segments. The
top face gets only one of its four edges.

**Root cause**
`cube_drawing_path()` builds the path as `[edges[0][0]] + [b for (a, b) in edges]`,
which assumes every edge in `cube_edges()` starts where the previous one ended.
The list breaks that chain (`(5, 1)` is followed by `(5, 6)`), so the path is
`[0, 1, 2, 3, 0, 4, 5, 1, 6, 2, 7, 3, 4]`: it draws the diagonals `1→6`, `2→7`,
`3→4` (14.1 cm) and never draws `4–7`, `5–6`, `6–7`.
The docstring's "all 12 edges without lifting the pen" is also impossible: every
cube vertex has odd degree (3), so no Eulerian path exists.

**Fix**
Pending. Use an explicit vertex order that retraces 3 edges (15 segments), e.g.
`0-1-2-3-0-4-5-6-7-4-5-1-2-6-7-3` (verified: 15 real edges, all 12 covered,
3 retraced), or lift the pen; add a test that every
consecutive pair is a real edge and all 12 edges are covered.
Details: [C3_DRAW_CUBE.md §6](C3_DRAW_CUBE.md#6-known-issues).

**Takeaway**
The tracking metric (`FK(q_des)` vs actual EE) was sub-millimetre, so it hid a
wrong *plan*. Validate the geometry of the plan separately from how well the
controller follows it.

### 7. `b1` trapezoidal: synchronized joints miss the goal (fixed 2026-10-23)

**Symptom**
`trapezoidal_trajectory` ended up to **11°** away from `q_end` on some joints;
with `computed_torque` in `b4` the arm visibly stopped short.

**Root cause**
To make all joints finish at `T_total`, non-limiting joints reused `v_max` with
`a_max`. That profile covers `v·(T − v/a)`, not `Δq`. Only the slowest joint
was correct. `test_endpoints` checked only the *start* point, so it passed.

**Fix**
Keep `a_max` and solve `Δq = v·(T − v/a)` for the cruise velocity:
`v = (aT − √(a²T² − 4aΔq)) / 2`. Also append a final sample at exactly
`T_total` (it is rarely a multiple of `dt`). Regression test:
`test_synchronized_joints_reach_goal`; `test_endpoints` now checks both ends.

**Takeaway**
An endpoint test must check *both* endpoints, and a trajectory test should
check consistency (`q` is the integral of `q̇`), not only limits.

---

## Quick start (after the fixes above)

```bash
source .venv/bin/activate
pytest lab2_ur5e/test -v
python lab2_ur5e/src/c3_draw_cube.py   # opens the MuJoCo viewer
```

Notes:
- The test folder is `test/`, not `tests/` as written in `lab2_navigation.md`.
- `c1_pick_and_place.py` writes its CSV output into `lab2_ur5e/docs/`.
