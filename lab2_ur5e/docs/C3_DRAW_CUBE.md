# C3 — How `c3_draw_cube.py` Works

The capstone of Lab 2: the UR5e end-effector traces a 10 cm cube in MuJoCo.
It chains IK (A4), trajectory generation (B1) and a feedforward position-servo
controller. This page explains the pipeline, what each number in the output
means, and the known limitations.

Run (from the repo root, with the venv; see [LESSONS.md §1](LESSONS.md)):

```bash
env -u PYTHONPATH LD_LIBRARY_PATH= .venv/bin/python lab2_ur5e/src/c3_draw_cube.py
```

---

## 1. Pipeline at a glance

```
1. Geometry     cube_vertices()          → 8 vertices (x, y, z)
                cube_drawing_path()      → visiting order, 13 points
2. IK           solve_cube_ik()          → 13 joint configurations q
3. Trajectory   build_cube_trajectory()  → quintic between consecutive q,
                                            2 s per segment, dt = 2 ms
4. Settle       3 s holding the first q  → servo reaches steady state
5. Simulate     every 2 ms: ctrl → mj_step → error metric → viewer overlay
```

| Step | Function | Uses |
|---|---|---|
| Geometry | `cube_vertices`, `cube_edges`, `cube_drawing_path` | — |
| IK | `solve_cube_ik` | `a4.ik_damped_least_squares` (Pinocchio) |
| Trajectory | `build_cube_trajectory` | `b1.quintic_trajectory` |
| Control + sim | `main` | MuJoCo position servos + `qfrc_bias` |

---

## 2. How one segment is drawn (vertex A → vertex B)

IK is solved **only at the two vertices**. Everything in between is
interpolated in **joint space**:

```
Task space:   p_A ●                                    ● p_B
                  │ IK                              IK │
                  ▼                                    ▼
Joint space:  q_A (6 angles) ── quintic, T = 2 s ──►  q_B (6 angles)
                           q(t) every dt = 2 ms
                                     │ forward kinematics
                                     ▼
Task space:   EE follows FK(q(t)), a curve from p_A to p_B
```

Each joint follows the same rest-to-rest quintic time scaling:

```
q_i(t) = q_A,i + Δq_i · s(t/T),   s(u) = 10u³ − 15u⁴ + 6u⁵,   Δq_i = q_B,i − q_A,i
```

Consequences:

- The EE passes **exactly** through both vertices (to the IK tolerance, 1e-4).
- Between vertices the EE path is **curved**, because FK is nonlinear. On this
  cube the real edges bow out by **2.2–4.7 mm** over 100 mm (computed with
  Pinocchio FK on `q_des`, before any tracking error).
- The robot **stops** at every vertex (q̇ = q̈ = 0 at both ends of a quintic).
- `q`, `q̇`, `q̈` are analytic, so the controller gets exact feedforward terms.

### Joint-space quintic vs Cartesian minimum jerk

`s(u)` above *is* the minimum-jerk profile; `b1.minimum_jerk_trajectory` uses
the same polynomial. The only difference is **what** is interpolated:

| | c3 (quintic in joint space) | Minimum jerk in task space + IK |
|---|---|---|
| Interpolated quantity | `q` (6 joint angles) | `p` (EE position) |
| IK calls per segment | **2** (the vertices) | **~1000** (every 2 ms) |
| EE path between vertices | Curved | Straight |
| `q̇`, `q̈` | Analytic | Finite differences or `J⁺ · ṗ` |
| Singularities mid-segment | Cannot break the motion | IK can fail on the way |

c3 chooses joint space for simplicity and robustness; the cost is the few-mm
bow per edge. For drawing tasks where straightness matters, use Cartesian
interpolation (see `b4_trajectory_sim.py --profile min_jerk_cartesian`).

---

## 3. IK details (`solve_cube_ik`)

- Target orientation is **fixed** to the EE orientation at `Q_HOME`, so the
  "pen" always points the same way.
- Each vertex is solved with DLS IK seeded from the **previous solution**, so
  consecutive configurations stay close and the arm never flips branch.
- If IK fails, a `WARNING` is printed but the **unconverged q is still used**.

---

## 4. Control law (main loop)

The Menagerie UR5e actuators are position servos:
`τ = Kp·(ctrl − q) − Kd·q̇`. c3 hides feedforward inside the setpoint:

```python
ctrl = q_des + qfrc_bias / Kp + Kd * qd_des / Kp
# ⇒ τ = Kp·(q_des − q) + Kd·(qd_des − q̇) + C(q, q̇)·q̇ + g(q)
```

- `qfrc_bias / Kp` supplies gravity + Coriolis, so the arm does not sag.
- `Kd·qd_des / Kp` turns servo damping into velocity tracking, so it does not lag.
- `qdd_des` is **not** used (no inertia feedforward).

Full derivation: [CONTROL_DYNAMICS_NOTES.md §2](CONTROL_DYNAMICS_NOTES.md#2-how-the-feedforward-in-c3-works).
History of this fix (133 mm → 0.088 mm RMS): [LESSONS.md §3](LESSONS.md).

The **settle phase** (1500 steps = 3 s) runs the same law with `qd_des = 0`
at the first waypoint, then resets `data.time` so the trajectory starts at 0.

---

## 5. What the printed numbers mean

| Output | Measured as | It tells you |
|---|---|---|
| `\|e_cart\|`, RMS / Max / Final Cartesian error | `‖FK_pin(q_des) − site_xpos(attachment_site)‖` | How well the **controller tracks the joint trajectory** |
| `\|tau\|_max` | `max \|qfrc_actuator\|` | Actuator load (limits: 150 / 28 N·m) |

⚠️ The Cartesian error is measured against `FK(q_des)`, i.e. against the
**curved joint-space path**, not against the straight cube edges. A
sub-millimetre result means "the controller follows the plan", not "the
drawing is a perfect cube". Geometric error (§2) and the path bug (§6) are
invisible in this metric.

---

## 6. Known issues

1. **The drawn shape is not a cube: 3 edges are replaced by face diagonals.**
   `cube_drawing_path` takes the *end* vertex of every entry in
   `cube_edges()`, assuming consecutive edges chain head-to-tail. They do not
   (e.g. `(5, 1)` is followed by `(5, 6)`), so the path is
   `[0, 1, 2, 3, 0, 4, 5, 1, 6, 2, 7, 3, 4]`:
   - drawn diagonals (14.1 cm): `1→6`, `2→7`, `3→4`
   - missing edges: `4–7`, `5–6`, `6–7` (top face has only one edge)

   In the viewer the red trail leaves the green wireframe at those three
   segments. See [LESSONS.md §6](LESSONS.md).
2. **A single continuous stroke cannot cover all 12 edges exactly once.**
   All 8 cube vertices have degree 3 (odd), so no Eulerian path exists. A
   continuous drawing must retrace at least 3 edges (15 segments minimum), or
   lift the pen. A valid order: `0-1-2-3-0-4-5-6-7-4-5-1-2-6-7-3`
   (retraces `4–5`, `1–2`, `6–7`).
3. Edges are slightly curved (§2), by design of joint-space interpolation.
4. IK failures are only warned about, not handled (§3).
