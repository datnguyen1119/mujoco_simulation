# Lab 3: Lessons Learned

## Bugs & Fixes

### 2026-08-13 — MuJoCo 3.11 removed `MjData.qM` and changed the `mj_fullM` signature
**Symptom:** After the environment moved to `mujoco==3.11.0`, three tests in `tests/test_dynamics.py` (`TestCrossValidation::test_cross_val_home` / `_random` / `_zeros`) failed with `AttributeError: 'mujoco._structs.MjData' object has no attribute 'qM'`, raised from `a1_dynamics_fundamentals.py:151`.
**Root cause:** MuJoCo 3.11 dropped the Python-visible packed sparse inertia buffer `MjData.qM`. `mj_fullM` was re-signatured at the same time: it is now `mj_fullM(model, data, dst)` — it reads the inertia straight off `MjData` and writes the dense `(nv, nv)` result into `dst` — instead of the legacy `mj_fullM(model, dst, data.qM)`. The `dst` argument moved from second to third position, so the old call is both an attribute error *and* an argument-order error. `MjData.M` still exists in 3.11 but is the packed sparse buffer (length 21 for `nv=6`), not a dense matrix, so it is not a drop-in replacement.
**Fix:** Added `mj_dense_mass_matrix(mj_model, mj_data)` to `src/lab3_common.py` as the single version-agnostic accessor. It probes `getattr(mj_data, "qM", None)`: when absent it calls the 3.11 form `mujoco.mj_fullM(mj_model, mj_data, dense)`, and when present it falls back to the legacy `mujoco.mj_fullM(mj_model, dense, qM)`. `cross_validate_mass_matrix()` in `a1_dynamics_fundamentals.py` now calls the helper. Cross-validation error is unchanged at `3.34e-05`, matching the number already published in `docs/a1_dynamics_fundamentals.md`, which confirms the new call returns the same matrix. Suite back to `34 passed`.
**Takeaway:** Simulator accessors are API surface that churns between minor versions. Wrap every raw MuJoCo struct field the lab depends on in a helper inside `lab<N>_common.py` — then a breaking upstream rename is a one-line fix in one file instead of a grep across every script. Prefer a `getattr` probe over a version-string comparison: it tests for the capability actually needed rather than for a number.

### 2026-03-17 — Lab 3 had to be migrated back onto the canonical hardware baseline
**Symptom:** Lab 3 was previously documented as complete on a simplified local UR5e path, which did not satisfy the project-wide requirement for Menagerie UR5e + Robotiq.
**Root cause:** Lab-level implementation drifted away from the project hardware lock.
**Fix:** Rebuilt the canonical path around Menagerie `universal_robots_ur5e` + mounted `robotiq_2f85`, updated the Pinocchio model path, reran validation, and updated the docs/media to match the final stack.
**Takeaway:** A lab is not complete until the code, tests, docs, and media all refer to the same hardware baseline.

### 2026-03-17 — Menagerie servo actuators required torque-to-control mapping
**Symptom:** The canonical MuJoCo UR5e stack does not expose direct torque motors the way the earlier simplified Lab 3 path did.
**Root cause:** Menagerie UR5e uses arm actuators with their own gain/bias model and control ranges.
**Fix:** Added torque-to-control mapping in `lab3_common.py` so desired arm torques are converted into actuator controls before each MuJoCo step.
**Takeaway:** Do not assume torque commands can be written directly when migrating to Menagerie assets. Check the actuator model first.

### 2026-03-17 — Pinocchio parity depended on the mounted Robotiq payload inertial being correct
**Symptom:** Dynamics parity and gravity compensation failed until the mounted gripper payload seen by Pinocchio matched the executed MuJoCo stack closely enough.
**Root cause:** The fixed end-effector payload in the URDF did not initially match the compiled MuJoCo scene inertial properties.
**Fix:** Updated the `ee_link` inertial in `models/ur5e.urdf` to the compiled-scene mass, COM, and inertia.
**Takeaway:** For cross-engine parity, the mounted payload is not optional bookkeeping. Its inertial parameters matter.

### 2026-03-17 — Contact detection had to include the full real end-effector contact set
**Symptom:** A narrow contact filter can miss the first real table contact on the canonical stack.
**Root cause:** The first-contact link is not guaranteed to be only the terminal tool body.
**Fix:** `get_ee_and_table_ids()` now includes `wrist_3_link` and the mounted Robotiq bodies, and the force-control validation was rerun on that basis.
**Takeaway:** Contact logic should reflect the real executed geometry, not an idealized tip-only assumption.

### 2026-03-17 — 6D impedance orientation error needed the large-angle-safe formulation
**Symptom:** The earlier skew-symmetric orientation error is only reliable near small rotations.
**Root cause:** That formulation is a local approximation and can misbehave near large-angle errors.
**Fix:** Switched the canonical impedance orientation error to `pin.log3(R_des @ R_cur.T)`.
**Takeaway:** Reuse the Lie-log orientation error in shared Cartesian controllers unless there is a strong reason not to.

### 2026-03-17 — Native MuJoCo OpenGL rendering was unavailable in this environment
**Symptom:** The standard MuJoCo renderer could not create an OpenGL context here.
**Root cause:** The required GL shared libraries are not available to this runtime.
**Fix:** The validation video still uses the real MuJoCo simulation state by projecting actual MuJoCo geom poses frame by frame and overlaying measured values.
**Takeaway:** Environment-specific rendering failures should not block controller validation, but the limitation should be documented explicitly.

### 2026-03-15 — Force sensor reads articulation forces, not contact forces
**Symptom:** `<force site="tool_site"/>` style sensing reported force even without contact.
**Root cause:** MuJoCo `<force>` sensors measure body-level constraint forces, not only contact forces.
**Fix:** Switched to `mj_contactForce()` on the relevant contact pairs.
**Takeaway:** Use `mj_contactForce()` for contact control logic.

### 2026-03-15 — Force control needs filtering and damping at stiff contact
**Symptom:** Raw PI contact control chatters at the table.
**Root cause:** Stiff contact plus integral action creates oscillatory force behavior without additional damping.
**Fix:** Retained EMA force filtering and Z-velocity damping in the canonical controller.
**Takeaway:** Stable force control needs both measurement smoothing and contact-direction damping.

### 2026-10-17 — Lab 3 cannot find the Menagerie models after the local folder rename
**Symptom:** 16 failed + 3 errors in the test suite, all from `ValueError: ParseXML: Error opening file '.../lab-2-Ur5e-robotics-lab/models/mujoco_menagerie/universal_robots_ur5e/ur5e.xml'`.
**Root cause:** Lab 3 reuses Lab 2's Menagerie checkout through a hard-coded folder name (`lab3_common.py:22`, `LAB2_MODELS_DIR`). Locally Lab 2 lives in `lab2_ur5e/`, not `lab-2-Ur5e-robotics-lab/`. The Lab 2 sparse checkout also lacked `robotiq_2f85/`, which Lab 3 attaches to the arm.
**Fix:** `LAB2_MODELS_DIR = PROJECT_ROOT / "lab2_ur5e" / "models"`; `git -C lab2_ur5e/models/mujoco_menagerie sparse-checkout add robotiq_2f85`. Suite back to `34 passed`.
**Takeaway:** A cross-lab path is a hidden dependency: renaming a lab folder breaks every later lab that points into it. Grep for the old folder name after any rename.

### 2026-10-17 — Segfault on exit after closing the passive viewer
**Symptom:** Scripts using `mujoco.viewer.launch_passive` (b1 `--interactive`, a2 `--viewer`) finish their work, then die with `Segmentation fault (core dumped)` (exit 139). Reproduced with a 10-line script using only MuJoCo, so not lab code. Faulthandler: main thread in `glfw.terminate`, viewer thread still in `viewer.py:_launch_internal`.
**Root cause:** The passive viewer renders on a daemon thread and registers `glfw.terminate` with `atexit`. When the window closes right before the script ends, the interpreter runs `glfw.terminate()` while the viewer thread is still destroying its window (MuJoCo 3.16, Linux/X11, NVIDIA 580).
**Fix:** `lab3_common.wait_for_viewer_shutdown(viewer)`: after the viewer closes, poll the handle's weakref `viewer._sim()` until it is `None` (teardown done, ~10 ms). Exit code 0 in 3/3 runs of the minimal script and in b1/a2.
**Takeaway:** A viewer on a background thread must be fully torn down before interpreter exit. Also: in a scripted smoke test, calling `viewer.close()` from a helper Python thread hung the process (cause not investigated); making `is_running()` return False instead works.

### 2026-10-18 — MuJoCo pinch site and Pinocchio EE frame differ by 90° about z
**Symptom:** In the b1_1 force probe, a 2 Nm torque about z with K_ori = 50 gave a measured rotation of 1.611 rad instead of the predicted 0.04 rad.
**Root cause:** The rotation was read from MuJoCo `site_xmat` of the pinch site and compared with a target taken from Pinocchio `oMf[ee_fid]`. The two frames share the origin (position parity 0 mm) but `R_pin.T @ R_site = Rz(−90°)`, so the error carried a constant π/2 offset (1.571 + 0.040).
**Fix:** Read EE orientation from Pinocchio FK (`get_ee_pose`) on the MuJoCo `qpos`, as the controller does. Measured rotation then matches τ/K_ori; suite `45 passed`.
**Takeaway:** Position parity does not imply frame parity. Never mix rotations from MuJoCo sites with rotations from Pinocchio frames; pick one source (Pinocchio, per the architecture principle).

## Final Validation Snapshot

- Full Lab 3 test suite: `34 passed`
- Max gravity mismatch: `8.01e-06`
- Max mass-matrix mismatch: `3.34e-05`
- Gravity hold max error: `8.91e-06 rad`
- Gravity perturbation final speed: `0.0134 rad/s`
- Hybrid force mean: `4.89 N`
- Hybrid force in-band rate: `99.96%`
- Hybrid force max XY error: `3.60 mm`
- Line trace in-band rate: `94.07%`
- Line trace max XY error: `1.70 mm`
