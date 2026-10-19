# Lab 3: TODO

## Canonical UR5e + Robotiq Path
- [x] Step R1: Replace the Lab 3 robot baseline with MuJoCo Menagerie UR5e + mounted Robotiq 2F-85
- [x] Step R2: Rebuild the MuJoCo scenes and matching Pinocchio model path on the canonical hardware stack
- [x] Step R3: Re-run FK, gravity, and mass-matrix parity checks on the canonical stack
- [x] Step R4: Re-qualify gravity compensation and impedance control on the canonical stack
- [x] Step R5: Fix and validate contact detection on the canonical stack, including all EE-adjacent first-contact links
- [x] Step R6: Re-run hybrid force control and constant-force line tracing on the canonical stack
- [x] Step R7: Expand tests and telemetry so reviewed risks are covered by regression checks
- [x] Step R8: Update README/docs/media so only the canonical stack is presented as Lab 3 completion

## Post-Completion Review Follow-Ups (2026-08-13)
- [x] Step F1: Fix the MuJoCo 3.11 `MjData.qM` / `mj_fullM` API break — added `mj_dense_mass_matrix()` to `lab3_common.py`, suite back to `34 passed` (see LESSONS.md)
- [x] Step F2: Write the blog post required by `plan/LAB_03.md` — `blog/lab3_blog_post.md`, "Position Control Pushes Through Walls"
- [x] Step F3: Document the orphaned demo pipeline — `src/record_lab3_demo.py` and its three MP4 outputs now appear in README and ARCHITECTURE.md

## Impedance Observer b1_1 (2026-10-18)
- [x] Step I1: b1 left unchanged; new `src/b1_1_impedance_observer.py` reuses its controller and adds terminal commands `goto` / `move` / `home` / `gains` / `force` / `pulse` / `show`, a force arrow, and measured vs predicted (K⁻¹·W) deflection in the overlay
- [x] Step I2: Headless `--probe` + `tests/test_impedance_observer.py` (parsers, F/K and τ/K deflection); suite `45 passed`. Probe 20 N down → 39.98 mm vs 40 mm predicted
- [x] Step I3: Tk slider panel (gains, 3D/6D, wrench hold/pulse, reset) + CLI `--gains` / `--mode` / `--no-panel`. Smoke run on the desktop: viewer and panel windows open; terminal `force 0 0 -20` → 40 mm (Medium), 10 mm with K_pos = 2000
- [ ] Step I4: Hands-on check of the sliders by the user

## Force Control Review Follow-Ups (2026-10-18)
Found while writing `docs/FORCE_CONTROL_NOTES.md`. None are blockers; the current demos pass.
- [ ] Step FC1: Add the missing `F_d` feedforward to the Z force law in `c1_force_control.py` (`compute_hybrid_torque` and `compute_hybrid_torque_joint`); the module docstring has `F_d +`, the code only has PI
- [ ] Step FC2: Hold EE orientation during contact in `c1`/`c2` (only the translational `J[:3]` is used, so 3 DOF are uncontrolled and undamped)
- [ ] Step FC3: Remove `abs()` at the end of `read_contact_force_z`; it hides sign errors in the geom-order handling
- [ ] Step FC4: Pass `ẋ_d = [0, 0, −descent_speed]` in the approach phase of `c1` and `c2` (currently ẋ_d = 0 → ~5 mm lag)
- [ ] Step FC5: Move the hard-coded `K_dz = 30.0` into `HybridGains`
- [ ] Step FC6: `generate_line_trajectory` uses `np.linspace(0, 1, n)` (step `1/(n−1)`) while indexing by `DT`; use `np.arange(n) * dt / duration`
- [ ] Step FC7: `c2` hold phase duplicates the logging block before `break`; the `main()` print hard-codes "1.5s / 6s / 1s"
- [ ] Step FC8: Decide on `solve_ik_xy` + `compute_hybrid_torque_joint` (unused, no tests): delete, or fix the Z conflict with a selection projection and add tests

## Current Focus
> Lab 3 is complete. No open implementation blockers remain inside this lab.

## Follow-On Dependency
> Labs 4 and 5 still need their own migration, review, and validation on the same hardware baseline.
