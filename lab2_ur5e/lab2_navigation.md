# Lab 2 Code Navigation Guide — UR5e 6-DOF

> Repo: `ozkannceylan/mujoco-robotics-lab` · Folder: `lab-2-Ur5e-robotics-lab/`
> Scope: `models/`, `src/`, `tests/` · Line numbers refer to `main` as analysed in October 2026.

---

## 1. The big picture in 30 seconds

```
Pinocchio  (models/ur5e.urdf)      = analytical brain: FK, Jacobian, M, C, g
MuJoCo     (models/lab_scene.xml)  = physics simulator: step, contact, render
ur5e_common.py                     = the only place that knows paths, constants, and how to load both models
```

The code follows a learning order, with three filename prefixes:

| Prefix | Meaning | Files |
|---|---|---|
| **A** | Foundations (kinematics, dynamics) | `a1` → `a5` |
| **B** | Motion and control | `b1` → `b3` |
| **C** | Integration and demos | `c1`, `c2`, `c3` |

Every `a*`, `b*` and `c*` file is **both a library and a demo script**: reusable functions at the top, a `main()` at the bottom that runs the demo.

---

## 2. Where to start

Recommended reading order:

1. `src/ur5e_common.py`: constants, DH table, model loaders
2. `models/lab_scene.xml` and `models/ur5e.urdf`: two descriptions of the same robot
3. `src/a1_model_setup.py`: how both models are loaded and their FK compared
4. Then `a2 → a3 → a4 → a5 → b1 → b2 → b3`, reading each file alongside its test
5. `src/c3_draw_cube.py`: the capstone, where the pieces come together

Tip: for each file, **read the module docstring first, then `main()` at the bottom**, then only the functions `main()` actually calls.

---

## 3. `models/`: robot and scene

```
models/
├── ur5e.urdf          ★ loaded by Pinocchio
├── lab_scene.xml      ★ loaded by MuJoCo
├── scene.xml            alias: just <include file="lab_scene.xml"/>
├── scene_link.xml       symlink → lab_scene.xml
├── assets               symlink → mujoco_menagerie/universal_robots_ur5e/assets
├── ur5e.xml             hand-written model — NOT used by any file (ignore it)
└── mujoco_menagerie/    gitignored, fetched by ./tools/setup_env.sh
```

### `ur5e.urdf`, for Pinocchio
- Masses and inertias are **copied from Menagerie's `ur5e.xml`**, so that Pinocchio and MuJoCo produce the same FK and dynamics. It is not UR's official URDF.
- Collision geometry uses simple cylinders.
- Lab 3 copies this file, and Labs 4, 5 and 6 share that copy.

### `lab_scene.xml`, for MuJoCo
- `<include>`s the Menagerie UR5e, then adds a floor, lights, a skybox, and two target spheres: `target_red` at (0.4, 0.15, 0.44) and `target_green` at (0.5, −0.15, 0.44).
- The spheres have `contype="0" conaffinity="0"`, so they are visual only and never collide.
- The header comment documents two traps:
  1. The scene must live outside the Menagerie folder, which is gitignored and disappears on a fresh clone.
  2. MuJoCo resolves `meshdir` relative to the **top-level file**, not the included one. That is why the `assets` symlink exists.

---

## 4. `src/`: file-by-file map

### 4.1 The two infrastructure files

#### `ur5e_common.py` (165 lines): configuration hub
| Line | Item | Note |
|---|---|---|
| 22–42 | `PROJECT_ROOT`, `MODELS_DIR`, `MEDIA_DIR`, `MJCF_SCENE_PATH`, `URDF_PATH` | Every path lives here |
| 47–69 | `GRAVITY`, `NUM_JOINTS`, `JOINT_LIMITS_*`, `VELOCITY_LIMITS`, `TORQUE_LIMITS` | Elbow limited to ±π, other joints ±2π |
| 76 | `class DHRow` | DH table used by `a2` |
| 96–103 | `Q_ZEROS`, `Q_HOME`, `LINK_MASSES` | Default configurations |
| 110, 116 | `mj_quat_to_pin`, `pin_quat_to_mj` | MuJoCo uses (w,x,y,z), Pinocchio uses (x,y,z,w) |
| 127 | `load_mujoco_model()` | Returns MuJoCo `(model, data)` |
| 144 | `load_pinocchio_model()` | Returns Pinocchio `(model, data)` |
| 162 | `get_mj_ee_site_id()` | End-effector site ID in MuJoCo |

#### `mujoco_sim.py` (163 lines): simulation wrapper
| Line | Item |
|---|---|
| 25 | `class RobotState`: q, qd, end-effector position… |
| 34 | `class UR5eSimulator`: load, step, read state, apply control, render |

Used by `a5`, `b2`, `c1`, `c2`.

---

### 4.2 Group A: foundations

#### `a1_model_setup.py` (207 lines): environment check
`check_dependencies` (35) → `check_model_files` (55) → `verify_mujoco` (71) → `verify_pinocchio` (111) → **`cross_validate_fk`** (144) → `main` (193)

Run this file first. If it fails, there is no point reading the others yet.

#### `a2_forward_kinematics.py` (274 lines): FK
| Line | Function | Role |
|---|---|---|
| 45 | **`dh_transform(alpha, a, d, theta)`** | 4×4 transform for one joint |
| 72 | **`fk_dh(q)`** | Chains the six transforms |
| 96 | `rotation_to_rpy` | R → roll/pitch/yaw |
| 157 | `validate_pin_vs_mujoco` | Compares FK between Pinocchio and MuJoCo |
| 229 | `note_on_dh_vs_urdf` | Explains why DH frames differ from URDF frames |

#### `a3_jacobian.py` (540 lines): Jacobian and singularities
| Line | Function | Role |
|---|---|---|
| 73 | **`geometric_jacobian`** | Built by hand from `z × (p_ee − p)` |
| 138 | **`pinocchio_jacobian`** | Taken from Pinocchio |
| 165 | **`numerical_jacobian`** | Finite differences, the third independent check |
| 221 | `manipulability` | `w = sqrt(det(J·Jᵀ))` |
| 240, 260 | `jacobian_determinant`, `detect_singularity` | Singularity detection |
| 310, 350 | `compute_manipulability_heatmap`, `save_heatmap_csv` | Manipulability map over q2 and q3 |
| 389–493 | `demo_*` | Demos called from `main` |

Key idea: **three ways of computing the Jacobian must agree.** This cross-validation pattern is used throughout the whole repo.

#### `a4_inverse_kinematics.py` (597 lines): numerical IK
| Line | Item | Role |
|---|---|---|
| 47 | `class IKResult` | q, success, iterations, error |
| 64 | **`ik_pseudoinverse`** | Δq = J⁺·Δx |
| 149 | **`ik_damped_least_squares`** | Δq = Jᵀ(JJᵀ + λ²I)⁻¹Δx, stable near singularities |
| 256 | `validate_ik_solution` | Re-checks the result with FK |
| 299 | `apply_ik_in_mujoco` | Sets q in MuJoCo to inspect the result |
| 341–444 | `generate_random_targets`, `run_benchmark`, `print_benchmark_summary` | Success rate and timing |

⚠️ **There is no analytical 8-solution IK**, even though the plan asks for one. This is a good gap to fill yourself.

#### `a5_dynamics.py` (399 lines): dynamics
| Line | Function | Pinocchio API |
|---|---|---|
| 59, 79 | `compute_mass_matrix`, `analyse_mass_matrix` | `crba` → M(q); symmetry, positive definiteness |
| 104 | `compute_gravity_vector` | `computeGeneralizedGravity` → g(q) |
| 125 | `compute_coriolis_matrix` | `computeCoriolisMatrix` → C(q, q̇) |
| 147 | `compute_rnea` | τ = M q̈ + C q̇ + g |
| 173 | `compute_aba` | q̈ from τ |
| 199 | **`cross_validate_gravity`** | Pinocchio g(q) vs MuJoCo `qfrc_bias` |
| 248 | `condition_number_sweep` | cond(M) across configurations |

---

### 4.3 Group B: motion and control

#### `b1_trajectory_generation.py` (355 lines)
| Line | Function | Property |
|---|---|---|
| 36 | `class JointTrajectoryPoint` | t, q, qd, qdd |
| 49 | `cubic_trajectory` | Zero start and end velocity |
| 81 | `quintic_trajectory` | Also zero start and end acceleration. The one most used by the `c*` files |
| 114 | `trapezoidal_trajectory` | Respects q̇_max and q̈_max |
| 214 | `minimum_jerk_trajectory` | s = 10τ³ − 15τ⁴ + 6τ⁵ |
| 241 | `multi_segment_trajectory` | Passes through several waypoints |

#### `b2_control_hierarchy.py` (383 lines): the control ladder
| Line | Function | Control law |
|---|---|---|
| 40 | `class ControlLog` | Records errors and torques |
| 54 | `pd_gravity_control` | τ = Kp·e + Kd·ė + g(q) |
| 84 | `computed_torque_control` | τ = M(q̈_d + Kp·e + Kd·ė) + Cq̇ + g |
| 111 | `task_space_impedance_control` | τ = Jᵀ(Kp·e_x + Kd·ė_x) + g |
| 171 | `osc_control` | τ = JᵀΛ(ẍ_d + …) + Jᵀμ + g, with Λ = (J M⁻¹ Jᵀ)⁻¹ |
| 236 | **`run_controller`** | Shared simulation loop for every controller |

Read the four controllers in line order: each one is an upgrade of the previous.

#### `b3_constraints.py` (367 lines)
| Line | Function |
|---|---|
| 43, 59 | `check_joint_limits`, `clamp_joint_positions` |
| 71 | `joint_limit_repulsion`: soft push away from limits |
| 102, 124 | `scale_velocity`, `scale_delta_q`: slow down while keeping direction |
| 144 | `saturate_torques` |
| 165, 200 | `check_self_collision`, `self_collision_score` |
| 233 | **`safe_command`**: applies every constraint to produce one safe command |

---

### 4.4 Group C: integration

#### `c3_draw_cube.py` (393 lines): ★ the capstone, worth reading
The end-effector draws the 12 edges of a cube.

```
cube_vertices (69) → cube_edges (91) → cube_drawing_path (111)
        → solve_cube_ik (127)          [DLS IK from a4 for the 8 vertices]
        → build_cube_trajectory (168)  [quintic from b1 for each edge]
        → main (247)                   [simulation + viewer + drawn trail]
```

Key technique: the Menagerie robot uses built-in **position servos**, so the command is augmented with:
```
ctrl = q_des + g(q)/Kp + Kd·qd_des/Kp     # gravity compensation + velocity feedforward
```
This brings tracking error down from about 133 mm to below 1 mm.

`c3_record_video.py` (168 lines) reuses `c3_draw_cube` to record a video into `media/`.

#### `c1_pick_and_place.py` (433 lines)
`solve_waypoints` (75) → `generate_pipeline_trajectory` (114) → **`run_pipeline`** (151) → `print_metrics` / `save_log_csv`
Combines a4 + b1 + b2 + b3. Also has `make_circle_waypoints` (281) for a circle-tracking demo.
⚠️ Despite the name, this file **has no gripper**: it only moves through waypoints. Actual grasping happens in Lab 5.

#### `c2_ros2_bridge.py` (159 lines)
A minimal ROS 2 node (`BridgeConfig`, `main`). It needs a sourced ROS 2 Humble install to run and has no MoveIt configuration. **Safe to skip while learning.**

---

### 4.5 Dependency graph

```
                       ur5e_common  ◄── (every file)
                            ▲
                       mujoco_sim   ◄── a5, b2, c1, c2

a4_inverse_kinematics ─┐
b1_trajectory ─────────┼──► c3_draw_cube ──► c3_record_video
                       │
b2_control_hierarchy ──┤
b3_constraints ────────┴──► c1_pick_and_place
```

`a1`, `a2`, `a3` and `a5` are independent leaves: they depend only on `ur5e_common` (plus `mujoco_sim` for `a5`), and no other file imports them.

---

## 5. `tests/`: 34 tests, mapped to `src/`

| Test file | Tests | Covers | What it checks |
|---|---|---|---|
| `test_ur5e_fk.py` | 6 | `a2`, `a3` | FK at home/zeros/random; geometric and numerical Jacobian vs Pinocchio; shape |
| `test_ur5e_ik.py` | 3 | `a4` | Pseudo-inverse and DLS recover a known q; FK round trip |
| `test_ur5e_dynamics.py` | 7 | `a5` | M symmetric, positive definite, correct shape; g non-zero; static RNEA = g; ABA = M⁻¹(τ − b) |
| `test_ur5e_trajectory.py` | 8 | `b1` | Cubic/quintic boundary conditions; trapezoidal respects limits; multi-segment |
| `test_ur5e_constraints.py` | 10 | `b3` | Joint limits, velocity scaling, torque saturation, repulsion |

- All use `unittest.TestCase`. Each file adds `src/` to `sys.path` itself.
- **Not tested:** `b2` (does a controller actually track?), `c1`, `c3`, `mujoco_sim`. Closed-loop behaviour is only checked by running the demos.

**Using the tests to learn:** read a test before the function it tests. The test tells you what the function *must* satisfy, for example "M must be symmetric and positive definite". Then read how the function achieves it.

---

## 6. Running things

```bash
# Once, from the repo root
./tools/setup_env.sh                 # fetches Menagerie, installs pin, mujoco…
export MUJOCO_GL=egl                 # only needed on a machine without a display

# Environment check
python3 lab-2-Ur5e-robotics-lab/src/a1_model_setup.py

# Individual topics
python3 lab-2-Ur5e-robotics-lab/src/a3_jacobian.py
python3 lab-2-Ur5e-robotics-lab/src/b2_control_hierarchy.py

# Capstone
python3 lab-2-Ur5e-robotics-lab/src/c3_draw_cube.py

# All tests
pytest lab-2-Ur5e-robotics-lab/tests/ -v
```

---

## 7. Common traps when reading or changing the code

| Trap | Where | Fix |
|---|---|---|
| `pip install pinocchio` installs the wrong package | environment | Use `pip install pin` |
| Missing meshes, `Error opening file '.../base_0.obj'` | `models/` | Run `setup_env.sh`; keep the `assets` symlink |
| Quaternion component order swapped | anywhere data passes between Pinocchio and MuJoCo | Always use `mj_quat_to_pin` / `pin_quat_to_mj` |
| DH frames differ from URDF frames | `a2` | Read `note_on_dh_vs_urdf` (line 229) |
| Position servos droop under gravity | `c3` | Add `qfrc_bias/Kp` and velocity feedforward |
| Editing `models/ur5e.xml` changes nothing | `models/` | That file is unused; the real scene is `lab_scene.xml` |

---

## 8. Suggested extensions (what the code is missing)

1. **Analytical IK for the UR5e** (8 solutions) in `a4`. Note that the UR5e does not have a spherical wrist; the closed-form solution relies on its three parallel axes (shoulder_lift, elbow, wrist_1).
2. **A closed-loop test for `b2`**: PD + gravity compensation should hold `Q_HOME` within 0.01 rad after 2 s of simulation.
3. **A test for `c3`**: cube-drawing RMS tracking error should stay below a set threshold.
4. **IK solution selection** by distance to the current q plus manipulability, as described in section 2.2 of the plan.
