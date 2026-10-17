# Lab 2 — Actuators, Feedforward and Dynamics

Background notes for the control bugs in [LESSONS.md](LESSONS.md) (bugs 3 and 4).
Robot: UR5e (MuJoCo Menagerie). Tools: MuJoCo (simulation) + Pinocchio (analytical model).

---

## 1. Two kinds of actuator

### 1.1 Position servo (Menagerie default, used in `c3`)

```xml
<option integrator="implicitfast"/>
<general gaintype="fixed" biastype="affine"
         gainprm="2000" biasprm="0 -2000 -400"
         ctrlrange="-6.2831 6.2831" forcerange="-150 150"/>   <!-- joints 1–3 -->
<!-- joints 4–6: gainprm="500" biasprm="0 -500 -100" forcerange="-28 28" -->
```
```
τ = gain·ctrl + bias₀ + bias₁·q + bias₂·q̇
  = Kp·(ctrl − q) − Kd·q̇          then clipped to ±forcerange
```

### 1.2 Torque motor (needed by `b2`, `c1`, Lab 3, Lab 8)

```xml
<motor gear="1" ctrlrange="-150 150"/>
```
```
τ = ctrl          then clipped to ±forcerange
```

### 1.3 Comparison

| | Position servo | Torque motor |
|---|---|---|
| What `ctrl` is | Target angle (rad) | Torque (N·m) |
| `ctrl = 0` | Pulls to angle 0 | Robot falls freely |
| Who carries gravity | The spring, but it sags without compensation | The controller, mandatory |
| Stiffness | Fixed in XML (Kp, Kd) | Free, set in code |
| Needs a dynamics model | Not required, but needed for accuracy | Required |
| Numerical stability | Very good (damping handled implicitly by `implicitfast`) | Gains must suit the time step |
| Suits | Trajectory tracking, drawing, simple pick-and-place | Impedance, OSC, force control, bipeds |
| Real-world analogue | Industrial arms taking joint angles (UR `servoj`) | Franka, KUKA iiwa, bipeds |

**Mental picture:**
- **Position servo:** you pull one end of a rubber band tied to the arm. The
  arm follows, but always lags and sags.
- **Torque motor:** you push the arm directly. Let go and it falls, but you
  decide the exact force at every instant.

**Bug 4 = code written for a torque motor, running on a position servo.**

### 1.4 Turning a servo into a torque motor at runtime

```python
m = sim.model
m.actuator_gainprm[:6, :] = 0;  m.actuator_gainprm[:6, 0] = 1     # τ = 1·ctrl
m.actuator_biasprm[:6, :] = 0                                       # remove spring + damper
m.actuator_biastype[:6] = mujoco.mjtBias.mjBIAS_NONE
m.actuator_ctrlrange[:6] = m.actuator_forcerange[:6]                # ±150 / ±28 N·m
```
Apply only to `b2`/`c1`. `c3` must keep the position servos.

---

## 2. How the feedforward in `c3` works

### 2.1 Why pure feedback always leaves an error

Robot equation: `M·q̈ + b = τ`, with `b = qfrc_bias = C·q̇ + g`.

With `ctrl = q_des` (no feedforward), let `e = q_des − q`:

```
M·q̈ + b = Kp·e − Kd·q̇
```

The robot needs torque for three things, and the servo can only produce torque
by **deflecting**:

| Robot needs | Required error | UR5e (Kp = 2000, Kd = 400) |
|---|---|---|
| Carry gravity `b` | `e = b/Kp` | b ≈ 60 N·m → 0.03 rad → ~25 mm at the tip* |
| Overcome damping while moving at `q̇_des` | `e = (Kd/Kp)·q̇_des` | **0.2 s lag** behind the trajectory |
| Accelerate inertia `M·q̈_des` | `e ≈ M·q̈_des/Kp` | Small for smooth trajectories |

\* Order-of-magnitude estimate, not a measurement.

### 2.2 Moving the spring's anchor point

```
ctrl = q_des  +  b/Kp  +  Kd·q̇_des/Kp
                 └ gravity ┘  └ damping ┘
```

Substituting into the servo:

```
τ = Kp·e + Kd·ė + b
    └ feedback ┘  └ feedforward ┘
```

Substituting into the robot equation, with `q̈ = q̈_des − ë`:

```
M·ë + Kd·ė + Kp·e = M·q̈_des
```

- Gravity **disappears** from the error dynamics.
- Damping now resists only the velocity **error** (`ė`), so the lag is gone.
- Only inertia `M·q̈_des` still causes error. Adding `M·q̈_des/Kp` to `ctrl`
  drives it to zero — and that is exactly computed torque.

### 2.3 Why not just raise Kp instead of feedforward

1. **Saturation:** with a large Kp, even a small error exceeds ±150 N·m.
2. **Oscillation:** `ζ = Kd / (2√(Kp·M))`. Raising Kp lowers ζ. Raising Kd too
   keeps the ratio Kd/Kp — i.e. the lag — unchanged.
3. **Numerical stability** at a 2 ms time step.
4. **Real robot:** amplifies noise, dangerously stiff on contact.

Feedforward does not depend on the error, so it has none of these drawbacks.

### 2.4 Structure: parallel sum, one closed loop

```
 q_des, q̇_des, q̈_des ──► [Model M̂, Ĉ, ĝ] ── τ_ff ──┐
          │                                         ▼
          └──►(−)──► [PD: Kp, Kd] ── τ_fb ──────► ( + ) ── τ ──► Robot ──┬──► q, q̇
               ▲                                                        │
               └─────────────── the only feedback loop ─────────────────┘
```

- Feedforward is **not a control loop**: it does not measure error and does not
  change the stability of the closed loop.
- It only makes the **right-hand side smaller**, i.e. less for feedback to fight.
- Different from **cascade** control (nested loops), where the outer loop's
  output is the inner loop's setpoint. Example in the series: Lab 8, DCM loop →
  CoM acceleration → whole-body QP.
- In `c3` the feedforward is "hidden" in the servo setpoint, because there is
  no other way to inject torque. Expanded, it is still `τ = Kp·e + Kd·ė + b`.
- Computed torque `τ = M(q)·(q̈_des + Kp·e + Kd·ė) + C·q̇ + g` is **not a pure
  sum**: feedback is multiplied by M(q). This is *feedback linearization*, which
  gives every joint the same error dynamics.

### 2.5 Levels of feedforward

| Level | Terms compensated | Example |
|---|---|---|
| None | — | `ctrl = q_des` → 133 mm |
| Gravity | g | PD + gravity comp (`b2`), Lab 3 |
| Gravity + Coriolis | C·q̇ + g (`qfrc_bias`) | `c3` (plus servo damping compensation) |
| Full | M·q̈ + C·q̇ + g | Computed torque (`b2`) |

---

## 3. Link to the equation of motion

```
M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ
```

This is a **law of physics**, not a controller. It can be used in two directions:

| Direction | Known | Computed | Who uses it |
|---|---|---|---|
| **Forward** dynamics | τ | q̈ = M⁻¹(τ − C·q̇ − g) | **Simulator**: MuJoCo, every step |
| **Inverse** dynamics | q̈ | τ | **Controller**: feedforward |

**Feedforward = inverse dynamics applied to the desired trajectory, with an
estimated model:**

```
τ_ff = M̂(q_des)·q̈_des + Ĉ(q_des, q̇_des)·q̇_des + ĝ(q_des)
```

Three differences from the physical equation:

1. **Desired vs actual:** physics runs on the real (q, q̇, q̈); feedforward uses
   (q_des, q̇_des, q̈_des). However far the robot deviates, feedback must make up.
2. **Estimated model (hats):** M̂, Ĉ, ĝ come from Pinocchio with the URDF and can
   be wrong (mass, friction, payload). Feedback corrects the mismatch `M − M̂`.
   This is why the repo always cross-validates Pinocchio against MuJoCo.
3. **Part of the equation can be used** (table 2.5).

In simulation, `c3` reads `qfrc_bias` **from MuJoCo itself**, so the model
matches 100 %. This is a "cheat" only possible in simulation. A real robot
does not expose `qfrc_bias`, so in practice you use Pinocchio.

```
      Controller (code)                                 Real world / MuJoCo
 τ_ff = M̂·q̈_des + Ĉ·q̇_des + ĝ   ← INVERSE dynamics    q̈ = M⁻¹(τ − C·q̇ − g)  ← FORWARD dynamics
 τ_fb = Kp·e + Kd·ė                  ── τ ──►         (real physics, real state)
```

If the prediction is perfect, the two sides cancel and the robot follows the
desired trajectory exactly.

---

## 4. Matching functions in Pinocchio (and MuJoCo)

### 4.1 Lookup table

| Quantity | Pinocchio | Result in | MuJoCo equivalent | Used for |
|---|---|---|---|---|
| **τ = M·q̈ + C·q̇ + g** (inverse) | `pin.rnea(model, data, q, v, a)` | return value, `data.tau` | `mj_inverse` → `data.qfrc_inverse` | **Feedforward**, computed torque |
| **q̈ from τ** (forward) | `pin.aba(model, data, q, v, tau)` | return value, `data.ddq` | `mj_forward` → `data.qacc` | Simulation, motion prediction |
| **M(q)** | `pin.crba(model, data, q)` | `data.M` (**upper triangle only**) | `mj_fullM(m, M, data.qM)` | Computed torque, OSC (Λ) |
| **M⁻¹(q)** | `pin.computeMinverse(model, data, q)` | `data.Minv` (upper triangle) | `mj_solveM` | OSC: Λ = (J M⁻¹ Jᵀ)⁻¹ |
| **g(q)** | `pin.computeGeneralizedGravity(model, data, q)` | `data.g` | (`qfrc_bias` when q̇ = 0) | Gravity compensation |
| **C(q, q̇)·q̇ + g(q)** | `pin.nonLinearEffects(model, data, q, v)` | `data.nle` | **`data.qfrc_bias`** | `c3`-style feedforward |
| **C(q, q̇) matrix** | `pin.computeCoriolisMatrix(model, data, q, v)` | `data.C` | — | Analysis, passivity proofs |

RNEA = *Recursive Newton–Euler Algorithm*, ABA = *Articulated Body Algorithm*,
CRBA = *Composite Rigid Body Algorithm*. RNEA and ABA are O(n); CRBA is O(n²)
since it returns an n×n matrix — all far faster than building and inverting
matrices.

### 4.2 Relations between the functions (also tests worth writing)

```python
import numpy as np, pinocchio as pin

pin.crba(model, data, q)
M = np.triu(data.M) + np.triu(data.M, 1).T           # symmetrize — CRBA fills only the upper triangle!

nle = pin.nonLinearEffects(model, data, q, v).copy()  # C·q̇ + g
g   = pin.computeGeneralizedGravity(model, data, q).copy()

# Special cases of RNEA
assert np.allclose(pin.rnea(model, data, q, v, np.zeros(nv)), nle)              # q̈ = 0      → C·q̇ + g
assert np.allclose(pin.rnea(model, data, q, np.zeros(nv), np.zeros(nv)), g)     # q̇ = q̈ = 0 → g

# RNEA is the inverse of ABA
qdd = pin.aba(model, data, q, v, tau).copy()
assert np.allclose(pin.rnea(model, data, q, v, qdd), tau)

# ABA = M⁻¹(τ − nle)
assert np.allclose(qdd, np.linalg.solve(M, tau - nle))

# Cross-check against MuJoCo
assert np.allclose(nle, mj_data.qfrc_bias[:6], atol=1e-2)
```

### 4.3 Feedforward variants in Pinocchio

```python
# (a) c3-style — position servo, model from Pinocchio instead of qfrc_bias
nle  = pin.nonLinearEffects(model, data, q, qd)
ctrl = q_des + nle / KP + KD * qd_des / KP

# (b) Full — position servo + inertia compensation
tau_ff = pin.rnea(model, data, q_des, qd_des, qdd_des)          # M·q̈_des + C·q̇_des + g
ctrl   = q_des + (tau_ff + KD * qd_des) / KP

# (c) Torque motor — PD + feedforward
tau = pin.rnea(model, data, q_des, qd_des, qdd_des) + Kp * (q_des - q) + Kd * (qd_des - qd)

# (d) Torque motor — computed torque (feedback linearization)
pin.crba(model, data, q); M = np.triu(data.M) + np.triu(data.M, 1).T
a_cmd = qdd_des + Kp * (q_des - q) + Kd * (qd_des - qd)
tau = M @ a_cmd + pin.nonLinearEffects(model, data, q, qd)
#     equivalent: pin.rnea(model, data, q, qd, a_cmd)
```

### 4.4 Traps when comparing Pinocchio and MuJoCo

| Trap | Consequence | Fix |
|---|---|---|
| `crba` fills only the **upper triangle** of `data.M` | M not symmetric, silently wrong results | Symmetrize as in 4.2 |
| **Armature**: Menagerie sets `armature="0.1"`, MuJoCo adds it to M's diagonal; the URDF has none | M̂ ≠ M, especially on the light wrist joints | Set `model.armature[:] = 0.1` in Pinocchio, or accept the mismatch |
| Quaternion (x,y,z,w) vs (w,x,y,z) | Only matters for floating bases (Lab 7+) | `mj_quat_to_pin` / `pin_quat_to_mj` |
| `qfrc_bias` is computed at the state of the last `mj_forward`/`mj_step` | Off by one step if read at the wrong time | Call `mj_forward` before reading |
| Joint damping/friction in the XML | Not included in RNEA | Compensate separately if significant |

---

## 5. Summary

1. **Read the actuator definition before writing a controller.** The same line
   `data.ctrl[:] = x` means an angle or a torque depending on the model.
2. **Feedback must be wrong to produce force. Feedforward uses the model to
   supply that force in advance.**
3. **Feedforward = inverse dynamics (RNEA) on the desired trajectory.**
   Simulation = forward dynamics (ABA). Both are two directions of the same
   equation `M·q̈ + C·q̇ + g = τ`.
4. **Feedforward + feedback is a parallel sum with one closed loop.**
   Feedforward does not change stability; it only makes feedback's job easier.
5. **In Pinocchio:** `rnea` (τ), `aba` (q̈), `crba` (M, remember to
   symmetrize), `nonLinearEffects` (= `qfrc_bias`), `computeGeneralizedGravity` (g).
