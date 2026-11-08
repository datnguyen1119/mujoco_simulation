"""Phase 3.4 — Interactive trajectory simulation in MuJoCo.

Pick a time-scaling profile (cubic, quintic, trapezoidal or Cartesian
minimum-jerk), a list of targets (joint angles or Cartesian positions) and a
controller, then watch the UR5e track it in the MuJoCo viewer.

Trajectories come from ``b1_trajectory_generation`` (and Pinocchio IK for
Cartesian targets); MuJoCo only simulates. Edit ``DEFAULTS`` below or
override anything from the command line.

Run (from the project root):
    python3 lab2_ur5e/src/b4_trajectory_sim.py
    python3 lab2_ur5e/src/b4_trajectory_sim.py --profile cubic --duration 3
    python3 lab2_ur5e/src/b4_trajectory_sim.py --profile trapezoidal --vmax-scale 0.3 \\
        --joint-target 0 -90 60 -60 -90 0 --joint-target -45 -90 45 -45 -90 0
    python3 lab2_ur5e/src/b4_trajectory_sim.py --profile min_jerk_cartesian \\
        --cart-target 0.4 0.15 0.44 --cart-target 0.5 -0.15 0.44
    python3 lab2_ur5e/src/b4_trajectory_sim.py --controller computed_torque --headless --plot
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from a4_inverse_kinematics import ik_damped_least_squares
from b1_trajectory_generation import (
    JointTrajectoryPoint,
    cubic_trajectory,
    quintic_trajectory,
    trapezoidal_trajectory,
)
from ur5e_common import (
    MEDIA_DIR,
    NUM_JOINTS,
    Q_HOME,
    VELOCITY_LIMITS,
    get_mj_ee_site_id,
    load_mujoco_model,
    load_pinocchio_model,
)

PROFILES = ("cubic", "quintic", "trapezoidal", "min_jerk_cartesian")
CONTROLLERS = ("servo", "servo_ff", "computed_torque")


# ---------------------------------------------------------------------------
# Configuration — edit here or override from the command line
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """All tunable parameters of one simulation run."""

    profile: str = "quintic"
    controller: str = "servo_ff"
    q_start_deg: np.ndarray = field(default_factory=lambda: np.rad2deg(Q_HOME))
    # Each target is either ("joint", 6 angles in deg) or ("cart", xyz in m).
    targets: list[tuple[str, np.ndarray]] = field(default_factory=lambda: [
        ("joint", np.array([-60.0, -70.0, 80.0, -100.0, -90.0, 0.0])),
        ("cart", np.array([0.4, 0.15, 0.44])),    # red marker in lab_scene
        ("cart", np.array([0.5, -0.15, 0.44])),   # green marker in lab_scene
    ])
    duration: float = 2.0          # seconds per segment (cubic / quintic / min-jerk)
    vmax_scale: float = 0.5        # trapezoidal: fraction of VELOCITY_LIMITS
    amax: float = 4.0              # trapezoidal: rad/s^2, all joints
    hold: float = 1.0              # seconds to hold the final pose
    kp: float = 400.0              # computed_torque gains (critically damped by default)
    kd: float = 40.0
    speed: float = 1.0             # viewer playback: 1.0 = real time
    headless: bool = False
    plot: Path | None = None


DEFAULTS = SimConfig()


# ---------------------------------------------------------------------------
# Trajectory construction
# ---------------------------------------------------------------------------


def _fk(model: pin.Model, data: pin.Data, ee_fid: int, q: np.ndarray) -> pin.SE3:
    """End-effector pose from Pinocchio FK."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_fid].copy()


def resolve_waypoints(
    cfg: SimConfig, model: pin.Model, data: pin.Data, ee_fid: int
) -> list[np.ndarray]:
    """Convert the configured targets into joint-space waypoints (rad).

    Cartesian targets keep the orientation of the previous waypoint and are
    solved with damped-least-squares IK. DLS is seed-sensitive, so several
    seeds are tried and the solution closest to the previous waypoint wins.
    """
    waypoints = [np.deg2rad(cfg.q_start_deg)]
    for kind, value in cfg.targets:
        if kind == "joint":
            waypoints.append(np.deg2rad(value))
            continue
        q_prev = waypoints[-1]
        R_keep = _fk(model, data, ee_fid, q_prev).rotation
        seeds = [q_prev, Q_HOME, np.deg2rad(cfg.q_start_deg)]
        solutions = [
            res.q for seed in seeds
            if (res := ik_damped_least_squares(
                model, data, ee_fid, value, R_keep, seed, max_iter=1000)).success
        ]
        if not solutions:
            raise ValueError(f"IK failed for Cartesian target {value} — is it reachable?")
        waypoints.append(min(solutions, key=lambda q: np.linalg.norm(q - q_prev)))
    return waypoints


def _min_jerk_cartesian_segment(
    q0: np.ndarray, q1: np.ndarray, T: float, dt: float,
    model: pin.Model, data: pin.Data, ee_fid: int,
) -> list[JointTrajectoryPoint]:
    """Straight-line EE motion with minimum-jerk timing, tracked by IK.

    Position follows p0 + s(t)(p1 - p0); orientation follows the geodesic
    R0 exp(s(t) log(R0^T R1)); s(t) = 10u^3 - 15u^4 + 6u^5. Joint velocity and
    acceleration are obtained by finite differences of the IK solutions.
    """
    X0, X1 = _fk(model, data, ee_fid, q0), _fk(model, data, ee_fid, q1)
    w = pin.log3(X0.rotation.T @ X1.rotation)
    n = max(2, int(np.ceil(T / dt)) + 1)
    times = np.linspace(0.0, T, n)
    qs = np.empty((n, NUM_JOINTS))
    q_seed = q0.copy()
    for k, t in enumerate(times):
        u = t / T
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        p = X0.translation + s * (X1.translation - X0.translation)
        R = X0.rotation @ pin.exp3(s * w)
        res = ik_damped_least_squares(model, data, ee_fid, p, R, q_seed, max_iter=300)
        if not res.success:
            raise ValueError(
                f"IK lost track of the Cartesian line at t={t:.3f}s, p={p.round(3)}. "
                "The straight line probably passes through a singularity (e.g. the "
                "wrist over the base axis) — move the targets or use a joint-space profile."
            )
        qs[k] = q_seed = res.q
    qs[-1] = q1  # land exactly on the waypoint (IK tolerance is 1e-4)
    qd = np.gradient(qs, times, axis=0)
    qdd = np.gradient(qd, times, axis=0)
    qd[[0, -1]] = 0.0
    qdd[[0, -1]] = 0.0
    return [JointTrajectoryPoint(t=t, q=qs[k], qd=qd[k], qdd=qdd[k]) for k, t in enumerate(times)]


def build_trajectory(
    cfg: SimConfig, waypoints: list[np.ndarray], dt: float,
    model: pin.Model, data: pin.Data, ee_fid: int,
) -> list[JointTrajectoryPoint]:
    """Chain one rest-to-rest segment per waypoint pair using the chosen profile."""
    traj: list[JointTrajectoryPoint] = []
    t_offset = 0.0
    for i, (q0, q1) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        if cfg.profile == "cubic":
            seg = cubic_trajectory(q0, q1, cfg.duration, dt)
        elif cfg.profile == "quintic":
            seg = quintic_trajectory(q0, q1, cfg.duration, dt)
        elif cfg.profile == "trapezoidal":
            seg = trapezoidal_trajectory(
                q0, q1, cfg.vmax_scale * VELOCITY_LIMITS,
                np.full(NUM_JOINTS, cfg.amax), dt,
            )
        elif cfg.profile == "min_jerk_cartesian":
            seg = _min_jerk_cartesian_segment(q0, q1, cfg.duration, dt, model, data, ee_fid)
        else:
            raise ValueError(f"unknown profile {cfg.profile!r}; choose from {PROFILES}")
        if i > 0:
            seg = seg[1:]  # first point duplicates the previous segment's last point
        traj.extend(
            JointTrajectoryPoint(t=p.t + t_offset, q=p.q, qd=p.qd, qdd=p.qdd) for p in seg
        )
        t_offset = traj[-1].t
    return traj


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


class Controller:
    """Turn a desired trajectory point into MuJoCo ``ctrl`` values.

    The Menagerie UR5e actuators are position servos:
    ``force = gain*ctrl + b0 + b1*q + b2*qd`` with ``b1 = -kp``, ``b2 = -kd``.

    - ``servo``: ``ctrl = q_des`` (built-in PD only → lags while moving and
      sags under gravity at rest).
    - ``servo_ff``: adds velocity feedforward (tracks ``qd_des``) and Pinocchio
      gravity compensation (removes the steady-state sag).
    - ``computed_torque``: Pinocchio inverse dynamics; the torque is mapped
      back through the actuator law so the servo outputs exactly that torque.
    """

    def __init__(self, cfg: SimConfig, mj_model: mujoco.MjModel,
                 pin_model: pin.Model, pin_data: pin.Data) -> None:
        self.mode = cfg.controller
        if self.mode not in CONTROLLERS:
            raise ValueError(f"unknown controller {self.mode!r}; choose from {CONTROLLERS}")
        self.kp, self.kd = cfg.kp, cfg.kd
        self.gain = mj_model.actuator_gainprm[:NUM_JOINTS, 0].copy()
        self.bias = mj_model.actuator_biasprm[:NUM_JOINTS, :3].copy()
        # Pinocchio's URDF has no rotor inertia; MuJoCo adds it as armature.
        self.armature = mj_model.dof_armature[:NUM_JOINTS].copy()
        self.pin_model, self.pin_data = pin_model, pin_data

    def torque_to_ctrl(self, tau: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Invert the affine actuator law for a desired joint torque."""
        b0, b1, b2 = self.bias.T
        return (tau - b0 - b1 * q - b2 * qd) / self.gain

    def __call__(self, ref: JointTrajectoryPoint, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if self.mode == "servo":
            return ref.q.copy()
        if self.mode == "servo_ff":
            kp_act, kd_act = -self.bias[:, 1], -self.bias[:, 2]
            g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            return ref.q + (kd_act * ref.qd + g) / kp_act
        M = pin.crba(self.pin_model, self.pin_data, q)
        M = np.triu(M) + np.triu(M, 1).T + np.diag(self.armature)
        nle = pin.nonLinearEffects(self.pin_model, self.pin_data, q, qd)
        a_cmd = ref.qdd + self.kp * (ref.q - q) + self.kd * (ref.qd - qd)
        return self.torque_to_ctrl(M @ a_cmd + nle, q, qd)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class SimLog:
    """Desired vs actual signals recorded every simulation step."""

    t: list[float] = field(default_factory=list)
    q_des: list[np.ndarray] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    qd_des: list[np.ndarray] = field(default_factory=list)
    qd: list[np.ndarray] = field(default_factory=list)
    ee: list[np.ndarray] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        """Stack every signal into a NumPy array."""
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def _draw_path(viewer, points: np.ndarray, targets: np.ndarray) -> None:
    """Draw the desired EE path (small green dots) and targets (red spheres)."""
    scn = viewer.user_scn
    scn.ngeom = 0
    eye = np.eye(3).flatten()
    for p, size, rgba in [*((p, 0.004, (0.2, 0.9, 0.3, 0.8)) for p in points),
                          *((p, 0.015, (0.9, 0.2, 0.2, 0.6)) for p in targets)]:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([size, 0, 0]), p, eye, np.array(rgba, dtype=np.float32))
        scn.ngeom += 1


def simulate(cfg: SimConfig) -> SimLog:
    """Build the trajectory, run it in MuJoCo and return the log."""
    mj_model, mj_data = load_mujoco_model()
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    dt = mj_model.opt.timestep
    site_id = get_mj_ee_site_id(mj_model)

    waypoints = resolve_waypoints(cfg, pin_model, pin_data, ee_fid)
    traj = build_trajectory(cfg, waypoints, dt, pin_model, pin_data, ee_fid)
    hold = JointTrajectoryPoint(t=traj[-1].t, q=traj[-1].q,
                                qd=np.zeros(NUM_JOINTS), qdd=np.zeros(NUM_JOINTS))
    refs = traj + [hold] * int(round(cfg.hold / dt))

    peak = np.max(np.abs([p.qd for p in traj]), axis=0) / VELOCITY_LIMITS
    print(f"Profile {cfg.profile}, controller {cfg.controller}: "
          f"{len(waypoints) - 1} segment(s), {traj[-1].t:.2f}s, "
          f"peak |qd|/limit = {peak.max():.2f}")
    if np.any(peak > 1.0):
        print(f"  WARNING: joints {np.where(peak > 1.0)[0].tolist()} exceed velocity limits")

    controller = Controller(cfg, mj_model, pin_model, pin_data)
    mj_data.qpos[:NUM_JOINTS] = waypoints[0]
    mj_data.ctrl[:NUM_JOINTS] = waypoints[0]
    mujoco.mj_forward(mj_model, mj_data)

    stride = max(1, len(traj) // 300)
    path_pts = np.array([_fk(pin_model, pin_data, ee_fid, p.q).translation for p in traj[::stride]])
    target_pts = np.array([_fk(pin_model, pin_data, ee_fid, q).translation for q in waypoints])

    log = SimLog()

    def step(ref: JointTrajectoryPoint) -> None:
        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()
        mj_data.ctrl[:NUM_JOINTS] = controller(ref, q, qd)
        mujoco.mj_step(mj_model, mj_data)
        log.t.append(mj_data.time)
        log.q_des.append(ref.q)
        log.qd_des.append(ref.qd)
        log.q.append(mj_data.qpos[:NUM_JOINTS].copy())
        log.qd.append(mj_data.qvel[:NUM_JOINTS].copy())
        log.ee.append(mj_data.site_xpos[site_id].copy())

    if cfg.headless:
        for ref in refs:
            step(ref)
    else:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            _draw_path(viewer, path_pts, target_pts)
            wall0 = time.perf_counter()
            for ref in refs:
                if not viewer.is_running():
                    break
                step(ref)
                viewer.sync()
                lag = mj_data.time / cfg.speed - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
            while viewer.is_running():  # keep the window open until the user closes it
                viewer.sync()
                time.sleep(0.02)

    _report(log, target_pts[-1])
    if cfg.plot is not None:
        plot_log(log, cfg)
    return log


def _report(log: SimLog, ee_goal: np.ndarray) -> None:
    """Print tracking statistics."""
    a = log.arrays()
    err = np.abs(a["q_des"] - a["q"])
    print(f"  max joint tracking error : {np.rad2deg(err.max()):.3f} deg")
    print(f"  final joint error        : {np.rad2deg(err[-1].max()):.4f} deg")
    print(f"  final EE error (MuJoCo)  : {1000 * np.linalg.norm(a['ee'][-1] - ee_goal):.2f} mm")


def plot_log(log: SimLog, cfg: SimConfig) -> Path:
    """Save desired vs actual joint position/velocity plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = log.arrays()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j in range(NUM_JOINTS):
        c = f"C{j}"
        axes[0].plot(a["t"], np.rad2deg(a["q_des"][:, j]), c, ls="--", lw=1)
        axes[0].plot(a["t"], np.rad2deg(a["q"][:, j]), c, lw=1.2, label=f"q{j + 1}")
        axes[1].plot(a["t"], a["qd_des"][:, j], c, ls="--", lw=1)
        axes[1].plot(a["t"], a["qd"][:, j], c, lw=1.2)
        axes[2].plot(a["t"], np.rad2deg(a["q_des"][:, j] - a["q"][:, j]), c, lw=1.2)
    axes[0].set_ylabel("q [deg]")
    axes[1].set_ylabel("qd [rad/s]")
    axes[2].set_ylabel("tracking error [deg]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(ncol=6, fontsize=8)
    axes[0].set_title(f"{cfg.profile} / {cfg.controller}  (dashed = desired)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    cfg.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.plot, dpi=120)
    plt.close(fig)
    print(f"  plot saved to {cfg.plot}")
    return cfg.plot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> SimConfig:
    """Build a SimConfig from DEFAULTS plus command-line overrides."""
    d = DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, default=d.profile)
    ap.add_argument("--controller", choices=CONTROLLERS, default=d.controller)
    ap.add_argument("--q-start", type=float, nargs=6, metavar="DEG",
                    default=list(d.q_start_deg), help="start joint angles in degrees")
    ap.add_argument("--joint-target", type=float, nargs=6, action="append", metavar="DEG",
                    help="add a joint-space target in degrees (repeatable)")
    ap.add_argument("--cart-target", type=float, nargs=3, action="append", metavar="M",
                    help="add a Cartesian EE target x y z in metres (repeatable)")
    ap.add_argument("--duration", type=float, default=d.duration,
                    help="seconds per segment (cubic/quintic/min_jerk_cartesian)")
    ap.add_argument("--vmax-scale", type=float, default=d.vmax_scale,
                    help="trapezoidal: fraction of the UR5e velocity limits")
    ap.add_argument("--amax", type=float, default=d.amax, help="trapezoidal: rad/s^2")
    ap.add_argument("--hold", type=float, default=d.hold, help="seconds to hold the end pose")
    ap.add_argument("--kp", type=float, default=d.kp, help="computed_torque P gain")
    ap.add_argument("--kd", type=float, default=d.kd, help="computed_torque D gain")
    ap.add_argument("--speed", type=float, default=d.speed, help="viewer playback speed")
    ap.add_argument("--headless", action="store_true", help="no viewer, just simulate")
    ap.add_argument("--plot", nargs="?", const=MEDIA_DIR / "b4_trajectory_sim.png", type=Path,
                    default=d.plot, help="save a tracking plot (optional path)")
    args = ap.parse_args(argv)

    # Targets from the CLI replace DEFAULTS.targets, keeping the order given.
    targets = list(d.targets)
    if args.joint_target or args.cart_target:
        targets = []
        flags = iter(a for a in (argv if argv is not None else sys.argv[1:])
                     if a in ("--joint-target", "--cart-target"))
        joints, carts = iter(args.joint_target or []), iter(args.cart_target or [])
        for flag in flags:
            if flag == "--joint-target":
                targets.append(("joint", np.array(next(joints))))
            else:
                targets.append(("cart", np.array(next(carts))))

    return SimConfig(
        profile=args.profile, controller=args.controller,
        q_start_deg=np.array(args.q_start), targets=targets,
        duration=args.duration, vmax_scale=args.vmax_scale, amax=args.amax,
        hold=args.hold, kp=args.kp, kd=args.kd, speed=args.speed,
        headless=args.headless, plot=args.plot,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    simulate(parse_args(argv))


if __name__ == "__main__":
    main()
