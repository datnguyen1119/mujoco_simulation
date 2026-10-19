# Lab 3 — What Kind of "Impedance" Is `b1`?

`src/b1_impedance_controller.py` is called an impedance controller, but its
control law

```
τ = Jᵀ(q)·[ K·x̃ − D·ẋ ] + g(q)          x̃ = x_des − x,  ẋ_des = 0
```

is **not the full impedance controller**. It is a **task-space PD with gravity
compensation** (also called *stiffness* or *compliance* control). It is the
special case of full impedance control obtained when the conditions in §3 hold.
In that case it is provably stable and gives the correct steady-state
stiffness, but **not** the designed transient behaviour.

---

## 1. Full impedance control (Hogan, 1985)

The goal of impedance control is to make the end-effector behave, *as seen by
the environment*, like a chosen mass–spring–damper:

```
Λ_d·ë + D·ė + K·e = F_ext          e = x − x_des = −x̃     (target impedance)
```

| Symbol | Meaning |
|---|---|
| `Λ_d` | desired (apparent) Cartesian inertia |
| `D`, `K` | desired damping and stiffness |
| `F_ext` | external force applied to the end-effector |

The robot itself obeys, in task space,

```
Λ(x)·ẍ + μ(x,ẋ)·ẋ + F_g(x) = F + F_ext,     τ = Jᵀ·F
Λ(x) = (J·M⁻¹·Jᵀ)⁻¹                           (Cartesian inertia)
```

Forcing the robot to follow the target impedance gives the full control law:

```
F = Λ(x)·ẍ_des + μ·ẋ  +  Λ(x)·Λ_d⁻¹·(D·x̃̇ + K·x̃)  +  (Λ(x)·Λ_d⁻¹ − I)·F_ext
τ = Jᵀ·F + g(q)
```

Ingredients it needs that `b1` does not have:

| Term | Needs |
|---|---|
| `Λ(x)·ẍ_des` | Cartesian inertia `Λ = (J M⁻¹ Jᵀ)⁻¹` and a desired acceleration |
| `μ·ẋ` | Cartesian Coriolis/centrifugal terms |
| `Λ·Λ_d⁻¹` | inertia shaping |
| `(Λ·Λ_d⁻¹ − I)·F_ext` | a **force/torque sensor** at the wrist |

This is the operational-space (OSC) structure from `b2_control_hierarchy.py` in
Lab 2, extended with an external-force term.

---

## 2. What `b1` implements

From `compute_impedance_torque` (`b1_impedance_controller.py:100`):

```python
F   = K_p @ error + K_d @ (xd_des - J @ qd)     # xd_des is always None → 0 in run_impedance_sim
tau = J.T @ F + g                               # g from pin.computeGeneralizedGravity
```

No `Λ`, no `μ`, no `F_ext`, no `ẍ_des`. In 6D mode, `error` stacks the position
error and the `log3` orientation error.

---

## 3. The conditions under which `b1` equals full impedance

Start from the full law in §1 and impose:

| # | Condition | Effect on the full law | Holds in `b1`? |
|---|---|---|---|
| C1 | **No inertia shaping:** `Λ_d = Λ(x)` (keep the robot's natural inertia) | `Λ·Λ_d⁻¹ = I`, so the `F_ext` term vanishes → **no force sensor needed** | Yes (by construction) |
| C2 | **Set-point regulation:** `ẋ_des = 0`, `ẍ_des = 0` | `Λ·ẍ_des` vanishes; `x̃̇ = −ẋ` | Yes — `run_impedance_sim` always passes `xd_des=None` |
| C3 | **Coriolis/centrifugal terms neglected:** drop `μ·ẋ` | Quadratic in velocity, zero at rest; stability still holds (see §4) | Yes (omitted) |
| C4 | **Exact gravity compensation:** `ĝ(q) = g(q)` | The `F_g` term cancels | Yes — Pinocchio vs MuJoCo gravity max error `8e-6` (`a1`) |
| C5 | **K, D symmetric positive definite** | Required for the energy argument in §4 | Yes — positive diagonals |
| C6 | **J full rank** along the motion (no singularity) | `Λ` exists; `Jᵀ` maps every task force | Yes near `Q_HOME`, not checked at runtime |
| C7 | **Task dimension = number of joints** (6D mode) | No unconstrained null space | Only in 6D mode — see §5 |

With C1–C4 the full law collapses to exactly the `b1` law:

```
F = D·(−ẋ) + K·x̃      τ = Jᵀ·F + g(q)
```

and the closed loop becomes

```
Λ(x)·ẍ + μ(x,ẋ)·ẋ + D·ẋ + K·(x − x_des) = F_ext
```

i.e. the robot's **own, configuration-dependent** inertia, plus the designed
spring and damper.

---

## 4. Why it is still stable (Takegaki & Arimoto, 1981)

Take the energy-like Lyapunov function (kinetic + virtual spring potential):

```
V = ½·q̇ᵀ·M(q)·q̇ + ½·x̃ᵀ·K·x̃
```

Using the skew-symmetry of `Ṁ − 2C` (the same property that lets us drop `μ`
in C3):

```
V̇ = −ẋᵀ·D·ẋ ≤ 0
```

So the controller only ever **removes** energy: it is passive and stable for
any SPD `K`, `D` (C5), without knowing `M` or `C`. With C6 and C7, LaSalle's
principle gives asymptotic convergence to `x_des`. This is the main reason the
simple form is popular: robust, model-light, and it only needs `g(q)`.

---

## 5. What you give up

### 5.1 The transient is not the one you designed

Because `Λ_d = Λ(x)` (C1), the effective damping ratio of each direction is

```
ζ = D / (2·√(K·λ))          λ = eigenvalue of Λ(x)
```

Measured on this lab's UR5e + Robotiq model (translational `Λ`, `K = 500 N/m`,
`D = 50 N·s/m`):

| Configuration | Eigenvalues of Λ (kg) | ζ range | ωₙ range (rad/s) | Max off-diagonal of Λ (kg) |
|---|---|---|---|---|
| `Q_HOME` | 1.63, 1.81, 5.88 | 0.46 – 0.87 | 9.2 – 17.5 | 1.62 |
| `Q_HOME` + random (1) | 1.36, 1.75, 8.65 | 0.38 – 0.96 | 7.6 – 19.2 | 2.93 |
| `Q_HOME` + random (2) | 1.77, 2.18, 5.67 | 0.47 – 0.84 | 9.4 – 16.8 | 1.15 |

The same gains give a **well-damped** response in one direction and a clearly
**under-damped** one in another, and the numbers change with the pose.
`Λ(x)` is not diagonal, so the axes are also **coupled**: pushing along x moves
the tip along y or z too.

With full impedance (`Λ_d` constant and diagonal, e.g. `2·I` kg), every
direction would have the same ζ and ωₙ, independent of the pose.

### 5.2 Tracking is not exact

If `xd_des ≠ 0` is passed (the code allows it), the missing `Λ·ẍ_des + μ·ẋ`
feedforward means the spring must deflect to accelerate the arm, so a moving
target is followed with lag. Same story as the 133 mm error in Lab 2's `c3`
before feedforward was added (see `lab2_ur5e/docs/CONTROL_DYNAMICS_NOTES.md §2`).

### 5.3 3D mode leaves the null space undamped

In 3D mode only 3 of the 6 joint DOF are constrained. `D` damps only the tip
velocity, so motions that leave the tip still (wrist reorientation, elbow
swing) have **no stiffness and no damping**: `V̇ = 0` along them, and LaSalle no
longer guarantees convergence in joint space. The fix is a null-space term:

```
τ = Jᵀ·F + g + (I − Jᵀ·J̄ᵀ)·(−D_null·q̇)       J̄ = M⁻¹·Jᵀ·Λ  (dynamically consistent pseudo-inverse)
```

6D mode (C7) does not have this issue, because J is square.

### 5.4 Apparent inertia cannot be changed

A light, safe-feeling end-effector (`Λ_d` < `Λ`) needs C1 removed, which in
turn needs a wrist F/T sensor to measure `F_ext`.

---

## 6. Summary

| | `b1` (task-space PD + g) | Full impedance |
|---|---|---|
| Control law | `Jᵀ(K·x̃ − D·ẋ) + g` | `Jᵀ[Λ·ẍ_des + μ·ẋ + Λ·Λ_d⁻¹(D·x̃̇ + K·x̃) + (Λ·Λ_d⁻¹ − I)·F_ext] + g` |
| Model needed | `g(q)`, `J` | `M`, `C`, `g`, `J` |
| Sensor needed | none | wrist F/T (if `Λ_d ≠ Λ`) |
| Steady-state stiffness | `K` ✔ | `K` ✔ |
| Transient | depends on pose and direction | as designed, uniform |
| Moving target | lags | tracked |
| Stability proof | passivity, model-free | needs an accurate model |

**In one sentence:** `b1` is full impedance control with the inertia left as
it is (C1), applied to a fixed target (C2), with Coriolis terms dropped (C3) and
exact gravity compensation (C4). It gets the stiffness right and is always
stable, but the transient response depends on the robot's pose.

---

## References

- N. Hogan, "Impedance Control: An Approach to Manipulation", *ASME J. Dyn. Sys., Meas., Control*, 1985.
- M. Takegaki, S. Arimoto, "A New Feedback Method for Dynamic Control of Manipulators", *ASME J. Dyn. Sys., Meas., Control*, 1981.
- O. Khatib, "A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation", *IEEE J. Robotics and Automation*, 1987.
- C. Ott, *Cartesian Impedance Control of Redundant and Flexible-Joint Robots*, Springer, 2008.
