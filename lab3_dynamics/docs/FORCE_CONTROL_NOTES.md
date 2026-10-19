# Lab 3 — Force Control Notes: From the Jacobian to Constant-Force Line Tracing

Study notes for the second half of Lab 3. They cover the kinematic and dynamic
background that force control relies on, how contact forces are read out of
MuJoCo, the hybrid position/force controller in `src/c1_force_control.py`, the
PD/PID theory behind its gains and desired velocities, and the capstone
`src/c2_line_trace.py`.

Companion note: `IMPEDANCE_CONTROL_NOTES.md` (what kind of impedance `b1` is).
Open code issues found while writing these notes are tracked in
`tasks/TODO.md`, not here.

---

## Part I — Kinematic and Dynamic Background

### 1. The Jacobian: one matrix, four uses

The Jacobian maps **joint velocities to end-effector (EE) velocity**:

```
ẋ = J(q)·q̇          ẋ = [v; ω] ∈ ℝ⁶,  q̇ ∈ ℝⁿ
```

For a revolute joint `i` with axis `zᵢ` through point `pᵢ`, column `i` is the
velocity the EE point `p` would have if only joint `i` turned at 1 rad/s:

```
Jᵢ = [ zᵢ × (p − pᵢ) ;  zᵢ ]
```

The same matrix, used in four directions:

| Direction | Formula | Needs an inverse? |
|---|---|---|
| Joint velocity → EE velocity | `ẋ = J·q̇` | No |
| EE force → joint torque | `τ = Jᵀ·F` | No |
| EE velocity → joint velocity (IK) | `q̇ = J⁻¹·ẋ` | Yes |
| Joint torque → EE force | `F = J⁻ᵀ·τ` | Yes |

The second row follows from power balance: `Fᵀ·ẋ = Fᵀ·J·q̇ = (Jᵀ·F)ᵀ·q̇ = τᵀ·q̇`.

The impedance controller `b1` and the hybrid controller `c1` only use the first
two rows (`v = J·q̇`, `τ = Jᵀ·F`). They never invert `J`, so they do not blow up
near a singularity. They only lose authority in the singular direction.

### 2. Jacobian shape for a 6-joint arm

**Rule: number of columns = joints that move the point; number of rows =
task dimensions you want to control.**

| Case | Shape | Consequence |
|---|---|---|
| EE frame, full pose | 6×6 | Invertible away from singularities |
| A point on link `k` (mid-chain) | 6×6, but the columns of joints `k+1 … 6` are zero | Only `k` directions are controllable at that point |
| One joint locked | 6×5 (drop that column) | One motion direction is lost; solve by least squares or drop one task row |
| Translation only (`J[:3]`, as in `c1`/`c2`) | 3×6 | The task is now redundant (see §3) |

### 3. Redundancy and the null space

When `J` is `m×n` with `n > m`, `ẋ = J·q̇` has infinitely many solutions:

```
q̇ = J⁺·ẋ + (I − J⁺·J)·q̇₀
```

- `J⁺·ẋ` is the minimum-norm solution.
- `(I − J⁺·J)·q̇₀` is motion in the **null space**: it does not move the EE,
  so it can serve a secondary goal (joint-limit avoidance, obstacle avoidance,
  posture).
- Several tasks with priorities can be stacked by projecting each one into the
  null space of the higher ones, or written as a QP.

Redundancy is **relative to the task**, not a property of the robot. A 6-joint
UR5e controlled with the translational Jacobian `J[:3]` (3×6) has 3 redundant
DOF. Two consequences in this lab:

- `compute_hybrid_torque` applies `τ = Jᵀ·F` with a 3×6 `J`. Nothing in the
  controller holds or damps the three orientation DOF, so the gripper can tilt.
  Only gravity compensation and the model's joint damping act on them.
- `solve_ik_xy` uses `np.linalg.lstsq`, which returns the minimum-norm step.
  It solves position only and does not keep the gripper pointing down.

A common fix is null-space damping: `τ_null = (I − Jᵀ·J⁺ᵀ)·(−D_n·q̇)`.

### 4. Estimating the EE force from joint torques

To estimate the external force with `F = J⁻ᵀ·τ`, `τ` must be the **external**
torque, not the motor torque:

```
τ_ext = M(q)·q̈ + C(q,q̇)·q̇ + g(q) − τ_motor
```

- At rest this reduces to `τ_ext = g(q) − τ_motor`.
- While moving, `q̈` is too noisy to use directly. A **momentum observer**
  avoids it. With `p = M(q)·q̇`:
  ```
  r(t) = K_O·[ p(t) − p(0) − ∫ (τ_motor + Cᵀ(q,q̇)·q̇ − g(q) + r) dt ]
  ```
  `r` converges to `τ_ext` as a first-order filter with bandwidth `K_O`.

Limits:

- Near a singularity `J⁻ᵀ` is ill-conditioned. Some EE forces produce no joint
  torque at all, so the joints cannot "see" them.
- The result is only correct if the force really acts at the frame used for
  `J`. A push on the forearm is wrongly attributed to the EE.

### 5. Torque saturation

Motors have hard limits. In this lab, `TORQUE_LIMITS = [150, 150, 150, 28, 28, 28] N·m`
(`lab3_common.py`), and `clip_torques` clips each joint independently, with no
warning.

What saturation breaks:

- **All stability guarantees.** The proofs assume the commanded torque is
  applied.
- **Force direction.** Clipping joints one by one changes the direction of
  `Jᵀ·F`, so the EE pushes in a different direction than commanded.
- **Gravity compensation.** If the clipped joint carries a large `g(q)`, part
  of the compensation is lost and the arm sags.
- **Integrators.** The integrator keeps accumulating while the output is
  saturated (windup), then overshoots when the saturation ends.

Mitigations:

- Clamp the task error, so large errors do not demand huge forces.
- Scale the task torque uniformly and keep gravity intact: pick the largest
  `s ∈ [0, 1]` such that `|g + s·τ_task| ≤ τ_max` holds for every joint.
- Use anti-windup on every integrator (as `c1` does).
- Put torque limits into a QP formulation as explicit constraints.

---

## Part II — Contact and Experiments

### 6. Reading contact forces from MuJoCo

**Pipeline per step.** Broad phase (bounding-box overlap) → filtering
(`contype`/`conaffinity` bitmasks, parent–child exclusion, `<exclude>` pairs)
→ narrow phase (exact geometry) → constraint solver.

**Each `mj_data.contact[i]` has:**

| Field | Meaning |
|---|---|
| `geom1`, `geom2` | The two geoms. **The order is not fixed**: the table can be either one. |
| `pos` | Contact point in world coordinates |
| `frame` | 3×3 contact frame stored row-wise; the first row is the normal, pointing from `geom1` to `geom2` |
| `dist` | Signed distance (negative = penetration) |

**`mj_contactForce(model, data, i, wrench)`** returns a 6D wrench **in the
contact frame**: `wrench[0]` is the normal force, `wrench[1:3]` the
tangential (friction) forces. To get the force in world coordinates:

```python
R = c.frame.reshape(3, 3)       # rows = contact axes in world
f_world = R.T @ wrench[:3]
```

**How `read_contact_force_z` (in `c1`) uses this:**

1. It keeps only contacts between the table geom and any geom whose **body**
   belongs to the EE (`wrist_3_link` or any `2f85_*` body). Filtering by body
   catches every gripper geom, including the first link that touches.
2. It projects the normal force onto world Z (`normal_force · normal[2]`).
3. It flips the sign depending on whether the table is `geom1` or `geom2`.
4. It sums over all contact points.

The final `abs()` is unnecessary. Worse, it hides a sign error in steps 2–3,
because a wrong sign would still give a positive number.

The raw signal is noisy, because contact forces jump between steps. Both `c1`
and `c2` low-pass filter it with an EMA (`α = 0.2`) before the controller uses
it.

**Debugging tip.** Print the body names of every contact pair. This is the
fastest way to find unexpected contacts, such as a link hitting the robot base
or the gripper fingers touching each other.

### 7. Perturbation tests

A perturbation is a deliberate push during the simulation, to see how the
controller reacts. In `b1`, the default is 30 N along X for 0.1 s, added to the
joint torques as `τ += Jᵀ·F`.

- A **short pulse** only shows the peak displacement and the recovery
  transient. It does **not** measure stiffness, because the response is
  dominated by inertia and damping.
- To check a stiffness `K`, push with a **constant** force and compare the
  steady-state deflection with `F/K`. `b1_1_impedance_observer.py --probe`
  does this: a 20 N push gives 39.98 mm, against 40 mm predicted.
- To model a real external push instead of adding to `τ`, use
  `mj_applyFT` to apply the force at the actual contact point. MuJoCo then
  treats it as an external load, and the controller has to react to it.

---

## Part III — Force Control

### 8. Hybrid position/force control (`c1`)

Idea (Raibert & Craig, 1981): split task space into directions that are
**position-controlled** and directions that are **force-controlled**, using a
selection matrix. In `c1`, XY is position-controlled and Z is force-controlled.

```
S_p = diag(1, 1, 0)        S_f = I − S_p = diag(0, 0, 1)

F_pos   = S_p·[ K_p·(x_d − x) + K_d·(ẋ_d − ẋ) ]
F_force = S_f·[ −(K_fp·e_f + K_fi·∫e_f dt) − K_dz·ż ]     e_f = F_d − F_measured
τ       = Jᵀ·(F_pos + F_force) + g(q)
```

How `compute_hybrid_torque` implements this:

| Step | Code |
|---|---|
| FK, translational `J` (3×6, `LOCAL_WORLD_ALIGNED`), `v = J·q̇`, `g(q)` | Pinocchio |
| `S_p` | Applied implicitly: the Z components of `pos_err` and `vel_err` are set to 0 |
| Force PI | `force_err = F_d − F_measured`, integrated with `DT` |
| Anti-windup | The integral is clamped so that `K_fi·∫e` stays within ±20 N |
| Sign | `F_z = −(…)`: pushing **down** (−Z) increases the measured contact force |
| Contact damping | `−K_dz·v_z` with `K_dz = 30`, against contact chatter |
| Output | `(τ, force_integral)`; the integral state is returned, so the function stays pure |

`run_hybrid_force_sim` runs two phases:

1. **Approach**: Cartesian impedance (`K = 800`, `D = 80`) toward a Z target
   that descends at 5 cm/s.
2. **Hybrid**: switches once, permanently, when the filtered force exceeds
   0.5 N.

The approach phase is needed because a force loop in free space has nothing
to push against. It just accelerates downward until something stops it.

### 9. Impedance control vs. hybrid force control

| | Impedance (`b1`) | Hybrid force (`c1`) |
|---|---|---|
| Control law | `F = K·(x_d − x) + D·(ẋ_d − ẋ)` on **every** axis | XY: position PD; Z: **PI on the force error** |
| What is regulated | The **relation** between force and deflection (a virtual spring) | The **contact force itself**, on Z |
| Force sensor | Not needed | Needed (`read_contact_force_z`) |
| Contact force | Indirect: `F ≈ K_z·(z_d − z_surface)`, so the surface position must be known | Direct: the integrator drives it to `F_d` without knowing the surface height |
| Free space | Stable, goes to the target | No Z target, so it descends until contact |
| Axis decoupling | No, one law for all axes | Yes, via `S_p` / `S_f` |
| Robustness in contact | Very robust, for stiff or soft environments | Sensitive to environment stiffness; needs filtering, damping and anti-windup |

In short: **impedance makes the robot a spring–damper, and the force is a side
effect of compressing it. Force control measures the force and corrects it
directly.** `c1` uses both: impedance to approach, hybrid control in contact.

### 10. The joint-space variant (`compute_hybrid_torque_joint`)

`c1` also contains an alternative that is **not called anywhere** (in `src/` or
`tests/`):

```
q_des = solve_ik_xy(xy, z, q)                     # position-only IK
τ     = K_p·(q_des − q) − K_d·q̇  +  Jᵀ·F_force  +  g(q)
```

The force branch is identical to §8. The position branch is a **joint-space PD**
on all six joints.

| | `compute_hybrid_torque` | `compute_hybrid_torque_joint` |
|---|---|---|
| Position loop | Cartesian, 3D | Joint space, 6D |
| Selection matrix | Yes (Z left to the force loop) | **No** (the PD also holds Z and orientation) |
| EE orientation | Not controlled | Held, because all joints are pulled to `q_des` |
| Null-space damping | None | `−K_d·q̇` on every joint |
| Input | `xy_des` | `q_des`, so IK is needed first |
| Unit of `K_p = 500` | N/m | N·m/rad: same number, different physics |

**The main problem is that two loops fight over Z.** `q_des` requires choosing
a height `z`. The joint PD then acts like a Z spring with stiffness about
`K_z ≈ (J·K_q⁻¹·Jᵀ)⁻¹_zz`, pulling the EE toward that height.

- If `z` equals the true contact height, there is no conflict.
- If `z` is δ above the surface, the PD pulls up with about `K_z·δ`. The
  integrator must cancel that, but it is clamped to ±20 N. With a stiff `K_z`,
  a few millimetres of δ is enough to saturate it, and the force never reaches
  5 N.
- If `z` is below the surface, the PD pushes down on its own, the force
  overshoots, and the integrator has to pull back.

This is closer to **parallel force/position control** (Chiaverini & Sciavicco,
1993) than to hybrid control: both loops act on Z, and the force loop only wins
through its integrator, as long as it is not saturated.

`solve_ik_xy` itself is undamped Newton iteration with a pseudo-inverse, step
0.5, and joint-limit clipping. Unlike `solve_dls_ik`, it ignores orientation,
is not damped near singularities, and returns `q` even when it did not
converge.

---

## Part IV — PD/PID Theory Behind the Gains

### 11. Regulation vs. tracking

| | Regulation (set-point) | Tracking (trajectory) |
|---|---|---|
| Goal | Reach a fixed `x_d` | Follow `x_d(t)` |
| Desired velocity | `ẋ_d = 0`, and that is **correct** | `ẋ_d(t) = dx_d/dt` |
| Control law | `F = K_p·e − K_d·ẋ` | `F = K_p·e + K_d·(ẋ_d − ẋ)` (+ feedforward) |

For regulation, PD plus gravity compensation is **globally asymptotically
stable** (Takegaki & Arimoto, 1981). The Lyapunov function is
`V = ½q̇ᵀM q̇ + ½eᵀK_p e`.

**Using `ẋ_d = 0` on a moving target causes a constant lag.** Suppose the
target moves at constant speed `v`, and the robot follows at steady state
(`ẋ = v`). The net force must then be about zero:

```
K_p·e + K_d·(0 − v) = 0    ⇒    e_ss = (K_d / K_p)·v
```

The D term stops damping the error and acts as viscous friction against the
motion. A position error has to build up to overcome it.

Example from `c1`'s approach phase: the Z target descends at 5 cm/s with
`ẋ_d = 0` and `K_d/K_p = 80/800`, so the lag is 5 mm. At 0.5 m/s it would be
5 cm. With `ẋ_d` included, the error obeys `K_p·e + K_d·ė ≈ 0` and decays to
zero instead.

**Going faster.** Speed comes from the trajectory `x_d(t)`, not from `ẋ_d`
alone. The three signals `x_d`, `ẋ_d`, `ẍ_d` must be **consistent** (each the
derivative of the previous one). In order of preference:

1. Generate a smooth trajectory (quintic, minimum-jerk) and feed `x_d`, `ẋ_d`.
2. Add acceleration feedforward when inertia matters. The full form is
   computed torque:
   ```
   τ = M(q)·(q̈_d + K_d·ė + K_p·e) + C(q,q̇)·q̇ + g(q)
   ```
   This makes the error dynamics `ë + K_d·ė + K_p·e = 0`, linear and
   independent of the configuration. The Cartesian version uses `Λ(x)`
   and `μ`.
3. Raise the gains only as a last resort. They are limited by torque
   saturation, contact stability and sensor noise.

**Rule: feedforward handles what is known in advance (trajectory, `F_d`,
gravity); feedback only corrects the deviation.**

### 12. What `ẋ_d` actually means

The derivative of the position error and the velocity error are the **same
thing**:

```
e = q_d − q    ⇒    ė = q̇_d − q̇
```

`q̇_d` is not a free choice; it is `dq_d/dt`. If `q_d` is constant, `q̇_d = 0`
because the derivative of a constant is zero.

**`q̇_d = 0` does not mean "the robot must not move". It means "the target is
not moving."** Ignoring inertia (a heavily damped robot), the PD law gives

```
K_p·e + K_d·(q̇_d − q̇) ≈ 0    ⇒    q̇ ≈ q̇_d + (K_p / K_d)·e
```

> **actual velocity = target velocity + a correction proportional to the
> remaining error**

| Case | Actual velocity |
|---|---|
| Fixed target (`q̇_d = 0`) | `q̇ = (K_p/K_d)·e`: it moves toward the target and slows down as it gets close |
| Moving target, `q̇_d` provided | `q̇ = q̇_d + (K_p/K_d)·e`: it moves with the target and closes the gap, so `e → 0` |
| Moving target, `q̇_d = 0` | It must lag by `e = (K_d/K_p)·v` to move at `v` |

With `K_p/K_d = 10 s⁻¹` and an initial error of 0.1 rad on a fixed target, the
robot starts at about 1 rad/s, and the error decays as `e₀·exp(−10·t)` (time
constant 0.1 s).

**Physical picture.** PD connects the robot to the target with a **spring**
(`K_p`) and a **damper** (`K_d`). The spring creates the motion; the damper
brakes it. `q̇_d` is the velocity of the spring's **anchor**. If the anchor is
moving but is declared as still, the damper resists the very motion the robot
needs to keep up.

**D is not velocity control.** The robot follows `v_des` alone only when
`K_p = 0`. With both terms, it follows the combination
`q̇_d + (K_p/K_d)·e`.

With inertia, the error obeys `M·ë + K_d·ė + K_p·e = 0` (given `M·q̈_d`
feedforward), with damping ratio `ζ = K_d / (2·√(K_p·M))`. The first-order
formula above is the high-damping approximation.

### 13. A non-zero desired velocity: three cases

| Command | Result |
|---|---|
| `q_d(t) = q₀ + v·t`, `q̇_d = v` (consistent) | Moves at `v` as long as the trajectory does. Real trajectories end with `q̇_d = 0`, so the robot stops because the trajectory says so. |
| `q_d` fixed, `q̇_d = v ≠ 0` (inconsistent) | Moves, then **stops past the target** at `q_d + (K_d/K_p)·v`, where the spring and the damper cancel. Example: `v = 0.5 rad/s`, `K_d/K_p = 0.1 s` → 0.05 rad offset. |
| `K_p = 0`, `q̇_d = v` | Pure velocity control: moves at `v` forever, and the position drifts. Useful for wheels, conveyors and joystick teleoperation. |

**In position control, position always wins.** The spring `K_p` never lets
the robot run away. So `q_d(t)` and `q̇_d(t)` must come **together** from one
trajectory.

### 14. PID in general

**D on error vs. D on measurement**

- `K_d·(ẋ_d − ẋ)` (D on error) is used with smooth trajectories.
- `−K_d·ẋ` (D on measurement) is used when the set-point **jumps**. With D on
  error, a step in `x_d` would make `ẋ_d` an impulse (*derivative kick*).
  Industrial PIDs use D on measurement by default, and `q̇_d = 0` in the code is
  exactly this form.

**I term during tracking.** The integral removes steady errors caused by
constant disturbances (gravity-model mismatch, friction). It does **not**
replace feedforward: on a moving target it has to charge and discharge,
causing phase lag and overshoot. It always needs anti-windup.

**The force loop has the same structure:**

| Position loop | Force loop in `c1` |
|---|---|
| `ẋ_d` / `ẍ_d` feedforward | `F_d` feedforward, **missing** in `c1` (the docstring has `F_d +`, the code only has PI, so the integrator must charge up to 5 N on its own) |
| D on measurement `−K_d·ẋ` | `−K_dz·ż`: the contact acts as a spring `F ≈ k_env·Δz`, so `Ḟ ≈ k_env·ż`. Damping the velocity is far less noisy than differentiating the force signal. |
| Low-pass filtered measurement | EMA on the contact force |

### 15. Estimating velocity from position only

Most real robots (including the UR5e) only have **joint encoders**. Velocity is
estimated from position. This works, but raw finite differences usually do
not.

**Why raw differences fail:**

- **Noise amplification.** Differentiation is `s` in the frequency domain:
  gain grows with frequency, which is where noise lives. With white noise of
  standard deviation `σ`:
  ```
  q̇̂_k = (q_k − q_{k−1}) / dt    ⇒    σ_q̇ = √2·σ / dt
  ```
  With `dt = 1 ms` and `σ = 10⁻⁴ rad`, the velocity noise is about
  0.14 rad/s, and it is then multiplied by `K_d`.
- **Encoder quantization.** With resolution `Δ`, velocity is quantized to
  multiples of `Δ/dt`:

  | Encoder | Δ (rad) | Velocity step at dt = 1 ms |
  |---|---|---|
  | 12-bit (4096 counts/rev) | 1.5·10⁻³ | 1.5 rad/s: unusable |
  | 20-bit | 6·10⁻⁶ | 6·10⁻³ rad/s: fine |

- **Filtering costs phase lag.** Every low-pass filter adds lag, which reduces
  the phase margin and limits how high `K_d` (and `K_p`) can go. The backward
  difference itself already lags by about half a sample. **This noise vs. lag
  trade-off is the core issue.**

**Differentiate `q`, not `q_err`.** Use `ė = q̇_d − q̇̂`, where `q̇_d` comes
analytically from the trajectory generator (exact, noise-free) and `q̇̂` is the
estimated measured velocity. Differentiating `q_err` would cause a derivative
kick on every set-point step.

**Practical estimators, from simple to complex:**

1. **Dirty derivative** (first-order filtered derivative, the standard in PID):
   ```
   D(s) = K_d·s / (1 + s/N)          N ≈ 2–20 × closed-loop bandwidth
   ```
   ```python
   qd_hat = a * qd_hat + (1 - a) * (q - q_prev) / dt    # a = exp(-N*dt)
   ```
2. **Savitzky–Golay** or a polynomial fit over a window: smoother, with a
   half-window delay.
3. **Model-based observers** (Luenberger, Kalman, high-gain): use the applied
   `τ` and the model `M·q̈ + C·q̇ + g = τ` to predict the velocity and correct
   it with the encoder. Less lag than pure filtering, but only as good as the
   model.
4. **Period measurement** (time between encoder edges): good at low speed,
   usually done in the drive hardware.

**Stability.** Kelly (1993) proved that for regulation,
`τ = K_p·(q_d − q) − K_d·ϑ + g(q)`, with `ϑ` a dirty derivative of `q`, is
still globally asymptotically stable using position measurements only. Fast
tracking with position-only feedback is harder and needs an observer
(e.g. Berghuis & Nijmeijer, 1993).

**In this lab**, MuJoCo provides `qvel` exactly, because it is an integrator
state, and every controller uses `mj_data.qvel`. This is easier than real
hardware. Replacing it with raw differences would still work in simulation
(no sensor noise), so the problem would not show up. To see it, add Gaussian
noise to `q` and quantize it to an encoder resolution.

---

## Part V — Capstone: Constant-Force Line Tracing (`c2`)

`c2_line_trace.py` adds no new controller. It reuses `compute_hybrid_torque`
and the helpers from `c1`, and adds a **trajectory generator** and a
**four-phase state machine**.

### 16. Quintic trajectory generation

`generate_line_trajectory` separates geometry from timing:

```
x_d(t) = x_start + s(τ)·(x_end − x_start),      τ = t / T ∈ [0, 1]
```

The time scaling is a quintic polynomial:

```
s(τ)  = 10τ³ − 15τ⁴ + 6τ⁵
s'(τ) = 30τ² − 60τ³ + 30τ⁴
```

It satisfies six boundary conditions: `s(0) = 0`, `s(1) = 1`, and zero velocity
and acceleration at both ends. The robot therefore starts and stops without a
jerk.

The velocity uses the chain rule, which is why there is a division by the
duration:

```
ẋ_d = s'(τ) · (1/T) · (x_end − x_start)
```

`ẋ_d` is computed **analytically**, not by differentiating the positions
(see §15). The velocity profile is bell-shaped, with a peak of
`v_max = 1.875·L/T`. For the default 50 mm line in 6 s, that is about
15.6 mm/s.

### 17. The four-phase state machine

```
 ┌──────────┐  fz > 0.5 N  ┌────────┐ after settle_time ┌────────┐ after trace_duration ┌────────┐ after post_time
 │0 Approach│ ───────────▶ │1 Settle│ ────────────────▶ │2 Trace │ ───────────────────▶ │3 Hold  │ ──────────▶ break
 └──────────┘              └────────┘                   └────────┘                      └────────┘
```

| Phase | Controller | XY target | `ẋ_d` | Purpose |
|---|---|---|---|---|
| 0 Approach | Cartesian impedance (`K = 800`, `D = 80`) | `xy_start`, Z descending at 5 cm/s | 0 (inconsistent with the descending Z, see §11) | Descend until contact |
| 1 Settle | `compute_hybrid_torque` | `xy_start` | `None` (= 0) | Stand still until the force settles at 5 N |
| 2 Trace | `compute_hybrid_torque` | `xy_traj[k]` | `xyd_traj[k]` | Draw the line while holding 5 N |
| 3 Hold | `compute_hybrid_torque` | `xy_end` | `None` (= 0) | Hold the end point, then exit |

Design points:

- **Settle exists for a reason.** Right after contact, the force oscillates and
  the integrator has not converged. Pulling sideways at that moment, with
  friction added, can make the EE bounce off or press too hard.
- **The 2 → 3 transition is smooth** because the quintic ends with
  `ẋ_d(T) = 0`. Passing `None` at that instant is consistent with the
  trajectory: the robot stops because the trajectory says so (§13, first case).
- **`force_integral` is not reset** between phases 1, 2 and 3. The desired force
  does not change, so the integral built up in phase 1 stays valid.
- The trajectory index is `int(t_since_trace / DT)`, clamped to the last
  sample. `total_duration = 15 s` is only a safety cap.

### 18. Friction, and why `c2` uses `K_p = 2000`

`c2` uses `K_p = 2000`, `K_d = 100`, four times stiffer than `c1`
(`500 / 50`), even though it already feeds `ẋ_d`.

The likely reason is **friction**. The table and most gripper geoms have a
sliding friction of 1.0 (some finger pads have 0.6–0.7). MuJoCo uses the
larger value of the pair, so `μ = 1.0`. Pressing with `F_N = 5 N`, the
tangential friction can reach `μ·F_N ≈ 5 N`. The position loop can only
overcome it with an error:

```
e ≈ F_friction / K_p      K_p = 500 → ~10 mm,    K_p = 2000 → ~2.5 mm
```

`ẋ_d` feedforward does not compensate friction. Only `K_p`, an integral term on
XY, or an explicit friction feedforward (for example
`F_fric = μ·F_N·v̂_d` along the direction of motion) can.

The lag that `ẋ_d` removes is small here, because the line is slow:
`(K_d/K_p)·v_max = (100/2000) × 15.6 mm/s ≈ 0.8 mm`. If the same 50 mm were
drawn in 0.5 s (`v_max ≈ 190 mm/s`), the lag without `ẋ_d` would be about 9 mm.
At that speed `ẋ_d`, and probably acceleration feedforward as well, start to
matter.

---

## Summary

| Topic | Key point |
|---|---|
| Jacobian | `ẋ = J·q̇` and `τ = Jᵀ·F` need no inverse; IK and force estimation do |
| Redundancy | Relative to the task: a translational `J` (3×6) leaves 3 DOF undamped |
| Torque saturation | Per-joint clipping changes the force direction; scale uniformly and keep `g` |
| Contact | `mj_contactForce` returns the wrench in the contact frame; geom order is not fixed |
| Hybrid vs. impedance | Impedance shapes the force–motion relation; force control regulates the measured force |
| `ẋ_d` | `ė ≡ q̇_d − q̇`; `q̇_d = 0` means a fixed target, and the robot still moves at `≈ (K_p/K_d)·e` |
| Tracking | Missing `ẋ_d` causes a lag of `(K_d/K_p)·v`; feedforward before higher gains |
| Velocity estimation | Raw differences amplify noise; use a dirty derivative or an observer |
| Line tracing | A quintic gives consistent `x_d`, `ẋ_d`; friction, not lag, sets `K_p` |

---

## References

- Raibert, M. H., & Craig, J. J. (1981). Hybrid position/force control of
  manipulators. *ASME J. Dynamic Systems, Measurement, and Control*, 103(2),
  126–133.
- Takegaki, M., & Arimoto, S. (1981). A new feedback method for dynamic
  control of manipulators. *ASME J. Dynamic Systems, Measurement, and Control*,
  103(2), 119–125.
- Hogan, N. (1985). Impedance control: An approach to manipulation.
  *ASME J. Dynamic Systems, Measurement, and Control*, 107(1), 1–24.
- Chiaverini, S., & Sciavicco, L. (1993). The parallel approach to
  force/position control of robotic manipulators. *IEEE Trans. Robotics and
  Automation*, 9(4), 361–373.
- Kelly, R. (1993). A simple set-point robot controller by using only position
  measurements. *Proc. 12th IFAC World Congress*.
- Berghuis, H., & Nijmeijer, H. (1993). A passivity approach to
  controller–observer design for robots. *IEEE Trans. Robotics and
  Automation*, 9(6), 740–754.
- De Luca, A., & Mattone, R. (2003). Actuator failure detection and isolation
  using generalized momenta. *Proc. IEEE ICRA* (momentum observer).
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). *Robotics:
  Modelling, Planning and Control*. Springer. Chapters 8–9.
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics*. Cambridge University
  Press. Chapter 9 (time scaling), Chapter 11 (force and hybrid control).
