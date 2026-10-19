"""Lab 3 — Phase 2: Cartesian Impedance Controller.

Implements task-space impedance control:
  F = K_p · Δx + K_d · Δẋ   (position + orientation)
  τ = J^T · F + g(q)

Supports translational-only (3D) and full 6D (position + orientation) modes.

Note: this is a task-space PD with gravity compensation, i.e. impedance control
without inertia shaping (Λ_d = Λ(x)), for a fixed target, with Coriolis terms
dropped. See docs/IMPEDANCE_CONTROL_NOTES.md for the conditions and trade-offs.

Usage:
    python b1_impedance_controller.py                # headless demos, saves plots
    python b1_impedance_controller.py --interactive  # drag the arm in the MuJoCo viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pinocchio as pin

from lab3_common import (
    DT,
    NUM_JOINTS,
    Q_HOME,
    MEDIA_DIR,
    apply_arm_torques,
    clip_torques,
    get_ee_pose,
    get_mj_ee_site_id,
    load_mujoco_model,
    load_pinocchio_model,
    wait_for_viewer_shutdown,
)


# ---------------------------------------------------------------------------
# Gains dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImpedanceGains:
    """Impedance controller gains.

    For translational-only mode, use 3x3 matrices.
    For full 6D mode, use 6x6 matrices (upper-left 3x3 = translation,
    lower-right 3x3 = orientation).

    The matrix size selects the mode in compute_impedance_torque. Each diagonal
    entry is the virtual spring (K_p) or damper (K_d) along one task axis:
    F_i = K_p[i,i]·Δx_i + K_d[i,i]·Δẋ_i. Equal entries give an isotropic spring;
    unequal ones an anisotropic spring (e.g. diag([2000, 2000, 100]) = stiff in
    x/y, soft in z). Must be symmetric positive definite (condition C5 in
    docs/IMPEDANCE_CONTROL_NOTES.md). Isotropic K still does not give the same
    damping ratio per axis: ζ = D / (2√(K·λ)) depends on the Cartesian inertia Λ.
    """
    # Defaults = "Medium": diag(Kx, Ky, Kz) in N/m and diag(Dx, Dy, Dz) in N·s/m.
    K_p: np.ndarray = field(default_factory=lambda: np.diag([500.0, 500.0, 500.0]))
    K_d: np.ndarray = field(default_factory=lambda: np.diag([50.0, 50.0, 50.0]))

    @staticmethod
    def make_6d(
        K_pos: float = 500.0,
        K_ori: float = 50.0,
        D_pos: float = 50.0,
        D_ori: float = 5.0,
    ) -> "ImpedanceGains":
        """Create 6D gains with scalar stiffness/damping for position and orientation.

        Shorthand for an isotropic 6x6 spring: the 2 stiffness scalars expand to
        6 diagonal entries, diag(K_pos, K_pos, K_pos, K_ori, K_ori, K_ori),
        i.e. the same stiffness on x/y/z and the same on Rx/Ry/Rz (same for D).

        Args:
            K_pos: Translational stiffness (N/m).
            K_ori: Rotational stiffness (Nm/rad).
            D_pos: Translational damping (N·s/m).
            D_ori: Rotational damping (Nm·s/rad).

        Returns:
            ImpedanceGains with 6x6 K_p and K_d.
        """
        #               └──── x, y, z (N/m) ───┘ └─── Rx, Ry, Rz (N·m/rad) ───┘
        K_p = np.diag([K_pos, K_pos, K_pos, K_ori, K_ori, K_ori])
        K_d = np.diag([D_pos, D_pos, D_pos, D_ori, D_ori, D_ori])
        return ImpedanceGains(K_p=K_p, K_d=K_d)


# ---------------------------------------------------------------------------
# Orientation error
# ---------------------------------------------------------------------------

def orientation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Compute orientation error between desired and current rotation matrices.

    Uses the Lie algebra log-map, which remains valid near 180 degree errors.

    Args:
        R_des: Desired rotation matrix [3x3].
        R_cur: Current rotation matrix [3x3].

    Returns:
        Orientation error vector in world frame (3,).
    """
    # pin.log3: SO(3) log map. Turns a rotation matrix into a 3-vector ω = θ·axis
    # (axis-angle). R_des @ R_cur.T is the rotation still missing to reach R_des,
    # expressed in the world frame, so log3 of it is the orientation error.
    return pin.log3(R_des @ R_cur.T)


# ---------------------------------------------------------------------------
# Impedance controller
# ---------------------------------------------------------------------------

def compute_impedance_torque(
    pin_model,
    pin_data,
    ee_fid: int,
    q: np.ndarray,
    qd: np.ndarray,
    x_des: np.ndarray,
    R_des: np.ndarray | None,
    xd_des: np.ndarray | None,
    gains: ImpedanceGains,
) -> np.ndarray:
    """Compute joint torques from Cartesian impedance control.

    Supports two modes based on gains dimensions:
      - 3x3 gains: translational-only (position control in 3D)
      - 6x6 gains: full 6D (position + orientation)

    Control law:
      F = K_p · error + K_d · velocity_error
      τ = J^T · F + g(q)

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: End-effector frame ID.
        q: Joint angles (6,).
        qd: Joint velocities (6,).
        x_des: Desired EE position (3,).
        R_des: Desired EE rotation [3x3]. Required for 6D mode, ignored for 3D.
        xd_des: Desired EE velocity (6,) or (3,) or None (zero).
        gains: ImpedanceGains with K_p, K_d.

    Returns:
        Joint torques (6,).
    """
    dim = gains.K_p.shape[0]
    is_6d = dim == 6

    # FK + Jacobian
    # forwardKinematics: world placement of every JOINT for this q → pin_data.oMi.
    pin.forwardKinematics(pin_model, pin_data, q)
    # updateFramePlacements: propagates oMi to every FRAME (links, ee_link, ...)
    # → pin_data.oMf. Frames are not updated by forwardKinematics on its own.
    pin.updateFramePlacements(pin_model, pin_data)
    # computeJointJacobians: per-joint Jacobian columns stored in pin_data.J.
    # Must be called before getFrameJacobian. (It also redoes forward
    # kinematics internally, so the forwardKinematics call above is redundant.)
    pin.computeJointJacobians(pin_model, pin_data, q)

    # oMf[ee_fid]: SE3 pose of the EE frame in the world ("o" = origin/world,
    # "M" = placement, "f" = frame) with .translation (3,) and .rotation (3x3).
    oMf = pin_data.oMf[ee_fid]
    x_cur = oMf.translation.copy()
    R_cur = oMf.rotation.copy()

    # getFrameJacobian: 6x6 Jacobian of the EE frame, rows = [linear(3), angular(3)].
    # LOCAL_WORLD_ALIGNED: velocity measured at the EE origin but with axes
    # aligned to the world frame, matching x_cur and the log3 error above.
    # (LOCAL would use EE-frame axes; WORLD would measure at the world origin.)
    J_full = pin.getFrameJacobian(
        pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    # Compute current EE velocity
    v_cur = J_full @ qd  # [linear(3), angular(3)]

    # Gravity compensation
    # computeGeneralizedGravity: joint torques g(q) that exactly hold the arm
    # against gravity (RNEA with q̇ = q̈ = 0) → pin_data.g. Same as a2.
    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    g = pin_data.g.copy()

    if is_6d:
        # 6D error
        pos_err = x_des - x_cur
        ori_err = orientation_error(R_des, R_cur)
        error = np.concatenate([pos_err, ori_err])

        # Desired velocity (default zero)
        if xd_des is None:
            xd_des_full = np.zeros(6)
        elif xd_des.shape[0] == 3:
            xd_des_full = np.concatenate([xd_des, np.zeros(3)])
        else:
            xd_des_full = xd_des

        vel_err = xd_des_full - v_cur

        # Impedance force (6D)
        F = gains.K_p @ error + gains.K_d @ vel_err

        # Joint torques
        tau = J_full.T @ F + g
    else:
        # 3D translational only
        pos_err = x_des - x_cur
        J_trans = J_full[:3, :]

        if xd_des is None:
            xd_des_3 = np.zeros(3)
        else:
            xd_des_3 = xd_des[:3]

        vel_err = xd_des_3 - v_cur[:3]

        F = gains.K_p @ pos_err + gains.K_d @ vel_err

        tau = J_trans.T @ F + g

    return tau


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_impedance_sim(
    x_des: np.ndarray,
    R_des: np.ndarray | None = None,
    gains: ImpedanceGains | None = None,
    q_init: np.ndarray | None = None,
    duration: float = 3.0,
    perturb_at: float | None = None,
    perturb_force: np.ndarray | None = None,
    perturb_duration: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Run an impedance control simulation to a Cartesian target.

    Args:
        x_des: Target EE position (3,).
        R_des: Target EE rotation [3x3]. If None, uses initial orientation.
        gains: Impedance gains. Defaults to K_p=500, K_d=50 (3D).
        q_init: Initial joint config. Defaults to Q_HOME.
        duration: Simulation duration (s).
        perturb_at: Time to apply perturbation (s).
        perturb_force: Cartesian force perturbation (3,) applied at EE.
        perturb_duration: Duration of perturbation pulse (s).

    Returns:
        Dictionary with keys: time, q, qd, tau, ee_pos, ee_pos_err,
        ee_ori_err (if 6D).
    """
    import mujoco

    if gains is None:
        gains = ImpedanceGains()
    if q_init is None:
        q_init = Q_HOME.copy()
    if perturb_force is None:
        perturb_force = np.array([30.0, 0.0, 0.0])

    is_6d = gains.K_p.shape[0] == 6

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)

    # Set initial config
    mj_data.qpos[:NUM_JOINTS] = q_init
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # If no target orientation, use initial
    if R_des is None and is_6d:
        _, R_des = get_ee_pose(pin_model, pin_data, ee_fid, q_init)

    n_steps = int(duration / DT)
    time_arr = np.zeros(n_steps)
    q_arr = np.zeros((n_steps, NUM_JOINTS))
    qd_arr = np.zeros((n_steps, NUM_JOINTS))
    tau_arr = np.zeros((n_steps, NUM_JOINTS))
    ee_pos_arr = np.zeros((n_steps, 3))
    ee_pos_err_arr = np.zeros((n_steps, 3))
    ee_ori_err_arr = np.zeros((n_steps, 3)) if is_6d else None

    for i in range(n_steps):
        t = i * DT
        time_arr[i] = t

        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()
        q_arr[i] = q
        qd_arr[i] = qd

        # Impedance torque
        tau = compute_impedance_torque(
            pin_model, pin_data, ee_fid, q, qd,
            x_des, R_des, None, gains,
        )

        # External perturbation via Jacobian transpose
        if perturb_at is not None and perturb_at <= t < perturb_at + perturb_duration:
            # Same Jacobian as in compute_impedance_torque (same q), recomputed
            # here; [:3, :] keeps the linear rows so J_trans.T maps a force to torques.
            pin.computeJointJacobians(pin_model, pin_data, q)
            J_trans = pin.getFrameJacobian(
                pin_model, pin_data, ee_fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            tau += J_trans.T @ perturb_force

        tau = clip_torques(tau)
        tau_arr[i] = tau

        apply_arm_torques(mj_model, mj_data, tau)
        mujoco.mj_step(mj_model, mj_data)

        # Record EE position
        ee_pos = mj_data.site_xpos[site_id].copy()
        ee_pos_arr[i] = ee_pos
        ee_pos_err_arr[i] = x_des - ee_pos

        if is_6d and ee_ori_err_arr is not None:
            _, R_cur = get_ee_pose(pin_model, pin_data, ee_fid, q)
            ee_ori_err_arr[i] = orientation_error(R_des, R_cur)

    result: Dict[str, np.ndarray] = {
        "time": time_arr,
        "q": q_arr,
        "qd": qd_arr,
        "tau": tau_arr,
        "ee_pos": ee_pos_arr,
        "ee_pos_err": ee_pos_err_arr,
    }
    if ee_ori_err_arr is not None:
        result["ee_ori_err"] = ee_ori_err_arr
    return result


# ---------------------------------------------------------------------------
# Interactive viewer mode
# ---------------------------------------------------------------------------

# (label, K_pos [N/m], D_pos [N·s/m]) — same presets as b2_compliance_demo.
# In 6D mode the rotational gains are K_pos/10 and D_pos/10, the ratio main() uses.
GAIN_PRESETS: Tuple[Tuple[str, float, float], ...] = (
    ("Soft", 100.0, 20.0),
    ("Medium", 500.0, 50.0),
    ("Stiff", 2000.0, 100.0),
)
TARGET_STEP = 0.02           # target displacement per key press (m)
SPRING_FORCE_FULL_RED = 30.0  # |K·e| (N) at which the spring line is fully red

# GLFW key codes. Letters and digits are avoided: the MuJoCo viewer already uses
# almost all of them for visualisation toggles, and the callback gets no modifiers.
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267
KEY_HOME, KEY_END, KEY_DELETE = 268, 269, 261

_TARGET_KEYS = {
    KEY_UP: np.array([TARGET_STEP, 0.0, 0.0]),
    KEY_DOWN: np.array([-TARGET_STEP, 0.0, 0.0]),
    KEY_LEFT: np.array([0.0, TARGET_STEP, 0.0]),
    KEY_RIGHT: np.array([0.0, -TARGET_STEP, 0.0]),
    KEY_PAGE_UP: np.array([0.0, 0.0, TARGET_STEP]),
    KEY_PAGE_DOWN: np.array([0.0, 0.0, -TARGET_STEP]),
}

_HELP_TEXT = (
    "Double-click a link, then Ctrl + right-drag: push it\n"
    "Up/Down: target ±x   Left/Right: target ±y\n"
    "PageUp/PageDown: target ±z\n"
    "Home: cycle stiffness   End: toggle 3D/6D\n"
    "Delete: reset to Q_HOME"
)


def _make_gains(preset_idx: int, is_6d: bool) -> ImpedanceGains:
    """Build 3x3 or 6x6 gains from a GAIN_PRESETS entry."""
    _, k_pos, d_pos = GAIN_PRESETS[preset_idx]
    if is_6d:
        return ImpedanceGains.make_6d(
            K_pos=k_pos, K_ori=k_pos / 10.0, D_pos=d_pos, D_ori=d_pos / 10.0
        )
    return ImpedanceGains(K_p=np.eye(3) * k_pos, K_d=np.eye(3) * d_pos)


def _add_sphere(scn, pos: np.ndarray, radius: float, rgba) -> None:
    """Append a sphere marker to a viewer scene."""
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]), pos, np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_segment(scn, p0: np.ndarray, p1: np.ndarray, width: float, rgba) -> None:
    """Append a capsule between two points to a viewer scene."""
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p0, p1)
    scn.ngeom += 1


def _add_axes(scn, pos: np.ndarray, R: np.ndarray, length: float, alpha: float) -> None:
    """Append an RGB (x, y, z) frame triad to a viewer scene."""
    for k in range(3):
        rgba = [0.0, 0.0, 0.0, alpha]
        rgba[k] = 1.0
        _add_segment(scn, pos, pos + length * R[:, k], 0.003, rgba)


def run_impedance_interactive(q_init: np.ndarray | None = None) -> None:
    """Run the impedance controller live in the MuJoCo viewer until it is closed.

    The target starts at the EE pose of q_init. Mouse perturbations from the
    viewer reach the arm as real external forces (data.xfrc_applied), so the
    controller sees a genuine F_ext, unlike the torque pulse in run_impedance_sim.

    Markers: green sphere = target, white sphere = EE, capsule = virtual spring
    (green → red with |K·e|). In 6D mode, faint triad = target orientation,
    solid triad = current orientation.

    Args:
        q_init: Initial joint config. Defaults to Q_HOME.
    """
    import mujoco
    import mujoco.viewer

    if q_init is None:
        q_init = Q_HOME.copy()

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()
    x_home, R_home = get_ee_pose(pin_model, pin_data, ee_fid, q_init)

    # Key presses arrive on the viewer thread: the callback only records them,
    # and the simulation loop applies them between steps.
    state = {"preset": 1, "is_6d": True, "offset": np.zeros(3), "reset": True}

    def on_key(keycode: int) -> None:
        if keycode in _TARGET_KEYS:
            state["offset"] = state["offset"] + _TARGET_KEYS[keycode]
        elif keycode == KEY_HOME:
            state["preset"] = (state["preset"] + 1) % len(GAIN_PRESETS)
        elif keycode == KEY_END:
            state["is_6d"] = not state["is_6d"]
        elif keycode == KEY_DELETE:
            state["reset"] = True

    sync_every = max(1, int(round((1.0 / 60.0) / DT)))  # ~60 Hz redraw
    print("Interactive impedance — close the viewer window to quit.\n" + _HELP_TEXT)

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=on_key,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = x_home
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 135.0

        step = 0
        saturated = False
        wall_start, sim_start = time.perf_counter(), 0.0

        while viewer.is_running():
            if state["reset"]:
                with viewer.lock():
                    mujoco.mj_resetData(mj_model, mj_data)
                    mj_data.qpos[:NUM_JOINTS] = q_init
                    mujoco.mj_forward(mj_model, mj_data)
                state["offset"] = np.zeros(3)
                state["reset"] = False

            is_6d = state["is_6d"]
            gains = _make_gains(state["preset"], is_6d)
            x_des = x_home + state["offset"]

            q = mj_data.qpos[:NUM_JOINTS].copy()
            qd = mj_data.qvel[:NUM_JOINTS].copy()
            tau = compute_impedance_torque(
                pin_model, pin_data, ee_fid, q, qd,
                x_des, R_home if is_6d else None, None, gains,
            )
            # compute_impedance_torque leaves pin_data.oMf at the current q.
            x_cur = pin_data.oMf[ee_fid].translation.copy()
            R_cur = pin_data.oMf[ee_fid].rotation.copy()

            tau_clipped = clip_torques(tau)
            saturated |= bool(np.any(tau_clipped != tau))
            apply_arm_torques(mj_model, mj_data, tau_clipped)
            mujoco.mj_step(mj_model, mj_data)
            step += 1

            if step % sync_every:
                continue

            # --- redraw (~60 Hz) ---
            err = x_des - x_cur
            spring_force = float(np.linalg.norm(gains.K_p[:3, :3] @ err))
            red = min(1.0, spring_force / SPRING_FORCE_FULL_RED)
            ori_err_deg = np.degrees(np.linalg.norm(orientation_error(R_home, R_cur)))

            with viewer.lock():
                scn = viewer.user_scn
                scn.ngeom = 0
                _add_sphere(scn, x_des, 0.015, [0.1, 0.9, 0.2, 0.6])
                _add_sphere(scn, x_cur, 0.008, [1.0, 1.0, 1.0, 1.0])
                _add_segment(scn, x_cur, x_des, 0.004, [red, 1.0 - red, 0.1, 0.9])
                if is_6d:
                    _add_axes(scn, x_des, R_home, 0.10, 0.35)
                    _add_axes(scn, x_cur, R_cur, 0.07, 1.0)

            label, k_pos, d_pos = GAIN_PRESETS[state["preset"]]
            viewer.set_texts([
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "Mode\nStiffness\nPosition error\nSpring force |K·e|\n"
                 "Orientation error\nTorque limit",
                 f"{'6D' if is_6d else '3D (orientation free)'}\n"
                 f"{label}  K={k_pos:.0f}  D={d_pos:.0f}\n"
                 f"{np.linalg.norm(err) * 1000:.1f} mm\n"
                 f"{spring_force:.1f} N\n"
                 f"{ori_err_deg:.1f} deg\n"
                 f"{'SATURATED' if saturated else 'ok'}"),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 _HELP_TEXT, None),
            ])
            saturated = False
            viewer.sync()

            # Keep simulation time in step with wall time; if the loop falls
            # behind, re-anchor instead of trying to catch up.
            lag = (mj_data.time - sim_start) - (time.perf_counter() - wall_start)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.1:
                wall_start, sim_start = time.perf_counter(), mj_data.time

    wait_for_viewer_shutdown(viewer)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run impedance controller demos: 3D position-only and 6D with orientation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interactive", action="store_true",
                        help="run live in the MuJoCo viewer and push the arm with the mouse")
    args = parser.parse_args()
    if args.interactive:
        run_impedance_interactive()
        return

    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Get EE position at Q_HOME as reference
    x_home, R_home = get_ee_pose(pin_model, pin_data, ee_fid, Q_HOME)
    print(f"EE at Q_HOME: pos={x_home}, R=\n{R_home}")

    # Target: 10cm forward from home
    x_target = x_home + np.array([0.0, 0.0, 0.10])

    # --- Demo 1: 3D translational impedance ---
    print("\n" + "=" * 60)
    print("Demo 1: 3D Translational Impedance (K=500, D=50)")
    print("=" * 60)
    # 3x3 gains → 3D mode: position only, orientation and null space are free.
    # One entry per axis [x, y, z]: 500 N/m = 5 N per cm of deflection.
    gains_3d = ImpedanceGains(
        K_p=np.diag([500.0, 500.0, 500.0]),
        K_d=np.diag([50.0, 50.0, 50.0]),
    )
    results_3d = run_impedance_sim(x_des=x_target, gains=gains_3d, duration=3.0)

    final_err = results_3d["ee_pos_err"][-1]
    final_err_mm = np.linalg.norm(final_err) * 1000
    print(f"  Final position error: {final_err_mm:.2f} mm")
    print(f"  Final error components: {final_err * 1000} mm")

    # --- Demo 2: 6D impedance with orientation ---
    print("\n" + "=" * 60)
    print("Demo 2: 6D Impedance (K_pos=500, K_ori=50)")
    print("=" * 60)
    # 6x6 gains → 6D mode: position + orientation, J is square (no free null space).
    # K_ori = 50 N·m/rad ≈ 0.87 N·m per degree of orientation error.
    gains_6d = ImpedanceGains.make_6d(K_pos=500.0, K_ori=50.0, D_pos=50.0, D_ori=5.0)
    results_6d = run_impedance_sim(
        x_des=x_target, R_des=R_home, gains=gains_6d, duration=3.0,
    )

    final_err_6d = results_6d["ee_pos_err"][-1]
    final_err_6d_mm = np.linalg.norm(final_err_6d) * 1000
    final_ori_err = np.linalg.norm(results_6d["ee_ori_err"][-1])
    print(f"  Final position error: {final_err_6d_mm:.2f} mm")
    print(f"  Final orientation error: {np.degrees(final_ori_err):.3f} deg")

    # --- Plot: 3D tracking ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = results_3d["time"]
    labels = ["X", "Y", "Z"]
    for j in range(3):
        axes[0].plot(t, results_3d["ee_pos"][:, j] * 1000, label=f"{labels[j]} actual")
        axes[0].axhline(x_target[j] * 1000, color=f"C{j}", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Position (mm)")
    axes[0].set_title("3D Impedance: EE Position Tracking")
    axes[0].legend(loc="right", fontsize=8)
    axes[0].grid(alpha=0.3)

    for j in range(3):
        axes[1].plot(t, results_3d["ee_pos_err"][:, j] * 1000, label=labels[j])
    axes[1].set_ylabel("Error (mm)")
    axes[1].set_title("Position Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for j in range(NUM_JOINTS):
        axes[2].plot(t, results_3d["tau"][:, j], label=f"J{j}", linewidth=0.8)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Joint Torques")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(MEDIA_DIR / "impedance_3d_tracking.png", dpi=150)
    plt.close(fig)

    # --- Plot: 6D tracking with orientation ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t6 = results_6d["time"]
    for j in range(3):
        axes[0].plot(t6, results_6d["ee_pos_err"][:, j] * 1000, label=labels[j])
    axes[0].set_ylabel("Position Error (mm)")
    axes[0].set_title("6D Impedance: Position Error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for j in range(3):
        axes[1].plot(t6, np.degrees(results_6d["ee_ori_err"][:, j]),
                     label=["Rx", "Ry", "Rz"][j])
    axes[1].set_ylabel("Orientation Error (deg)")
    axes[1].set_title("6D Impedance: Orientation Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for j in range(NUM_JOINTS):
        axes[2].plot(t6, results_6d["tau"][:, j], label=f"J{j}", linewidth=0.8)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(MEDIA_DIR / "impedance_6d_tracking.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {MEDIA_DIR}/")


if __name__ == "__main__":
    main()
