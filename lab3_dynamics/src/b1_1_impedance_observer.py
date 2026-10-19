"""Lab 3 — Phase 2 (observer): drive the b1 impedance controller by hand.

Reuses the controller from b1_impedance_controller unchanged and adds, on top
of the b1 viewer:
  - target pose commands typed in the terminal (goto / move / home),
  - custom gains (gains K_pos D_pos K_ori D_ori),
  - an external wrench pushed on the EE along a chosen direction
    (force / pulse), with the measured deflection shown next to the
    static prediction Δx = K_pos⁻¹·F, Δθ = K_ori⁻¹·τ,
  - a slider panel (Tk) for gains, mode and the external force.

Usage:
    python b1_1_impedance_observer.py                          # viewer + panel + terminal
    python b1_1_impedance_observer.py --gains 1000 150 300 30  # start with custom gains
    python b1_1_impedance_observer.py --mode 3d --no-panel     # position only, no panel
    python b1_1_impedance_observer.py --probe 0 0 -20          # headless: hold a force, print Δx
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pinocchio as pin

from b1_impedance_controller import (
    _TARGET_KEYS,
    GAIN_PRESETS,
    KEY_DELETE,
    KEY_END,
    KEY_HOME,
    SPRING_FORCE_FULL_RED,
    ImpedanceGains,
    _add_axes,
    _add_segment,
    _add_sphere,
    compute_impedance_torque,
    orientation_error,
)
from lab3_common import (
    DT,
    NUM_JOINTS,
    Q_HOME,
    apply_arm_torques,
    clip_torques,
    get_ee_pose,
    get_mj_ee_site_id,
    load_mujoco_model,
    load_pinocchio_model,
    wait_for_viewer_shutdown,
)


FORCE_ARROW_SCALE = 0.005  # arrow length per newton (m/N): 20 N → 10 cm

_HELP_TEXT = (
    "Double-click a link, then Ctrl + right-drag: push it\n"
    "Up/Down: target ±x   Left/Right: target ±y\n"
    "PageUp/PageDown: target ±z\n"
    "Home: cycle stiffness   End: toggle 3D/6D\n"
    "Delete: reset to Q_HOME\n"
    "Terminal: goto / move / force / pulse / gains (help)"
)

_CONSOLE_HELP = (
    "Commands (world frame; metres, degrees, newtons, newton-metres, seconds):\n"
    "  goto x y z [roll pitch yaw]     absolute target; orientation kept if omitted\n"
    "  move dx dy dz [dr dp dy]        shift the current target; rotation about world axes\n"
    "  home                            target back to the start pose (arm is not reset)\n"
    "  force fx fy fz [tx ty tz]       hold an external wrench on the EE\n"
    "  force off                       remove it\n"
    "  pulse fx fy fz [tx ty tz] T     apply the wrench for T seconds\n"
    "  gains K_pos D_pos K_ori D_ori   custom gains (Home key returns to the presets)\n"
    "  show                            print target, EE pose, wrench and deflection\n"
    "  help                            this text"
)


# ---------------------------------------------------------------------------
# Pure helpers (no simulation state)
# ---------------------------------------------------------------------------

def gains_from_scalars(
    k_pos: float, d_pos: float, k_ori: float, d_ori: float, is_6d: bool
) -> ImpedanceGains:
    """Build 3x3 (translation only) or 6x6 gains from scalar stiffness/damping.

    Args:
        k_pos: Translational stiffness (N/m).
        d_pos: Translational damping (N·s/m).
        k_ori: Rotational stiffness (Nm/rad), unused in 3D.
        d_ori: Rotational damping (Nm·s/rad), unused in 3D.
        is_6d: True for position + orientation, False for position only.

    Returns:
        ImpedanceGains with 3x3 or 6x6 diagonal K_p, K_d.
    """
    if is_6d:
        return ImpedanceGains.make_6d(K_pos=k_pos, K_ori=k_ori, D_pos=d_pos, D_ori=d_ori)
    return ImpedanceGains(K_p=np.eye(3) * k_pos, K_d=np.eye(3) * d_pos)


def preset_gains(preset_idx: int, is_6d: bool) -> ImpedanceGains:
    """Gains of a GAIN_PRESETS entry; rotational gains are the translational / 10, as in b1."""
    _, k_pos, d_pos = GAIN_PRESETS[preset_idx]
    return gains_from_scalars(k_pos, d_pos, k_pos / 10.0, d_pos / 10.0, is_6d)


def rpy_deg(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (deg) of a rotation matrix, R = Rz(yaw)·Ry(pitch)·Rx(roll)."""
    return np.degrees(pin.rpy.matrixToRpy(R))


def _parse_numbers(cmd: str, args: list[str]) -> np.ndarray:
    try:
        return np.array([float(a) for a in args])
    except ValueError:
        raise ValueError(f"'{cmd}' arguments must be numbers") from None


def apply_target_command(
    line: str,
    x_des: np.ndarray,
    R_des: np.ndarray,
    x_home: np.ndarray,
    R_home: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the new (x_des, R_des) after a `goto`, `move` or `home` command.

    Positions are in metres and angles in degrees, both in the world frame.
    Orientation uses roll-pitch-yaw, R = Rz(yaw)·Ry(pitch)·Rx(roll).

    Args:
        line: Command text, e.g. "goto 0.1 0.5 0.4 180 0 0" or "move 0 0 0.05".
        x_des: Current target position (3,).
        R_des: Current target rotation [3x3].
        x_home: Start EE position (3,), used by `home`.
        R_home: Start EE rotation [3x3], used by `home`.

    Returns:
        New target position (3,) and rotation [3x3].

    Raises:
        ValueError: Unknown command or wrong number / type of arguments.
    """
    parts = line.split()
    if not parts:
        raise ValueError("empty command")
    cmd, args = parts[0].lower(), parts[1:]

    if cmd == "home":
        if args:
            raise ValueError("'home' takes no arguments")
        return x_home.copy(), R_home.copy()

    if cmd not in ("goto", "move"):
        raise ValueError(f"unknown command '{cmd}' (type 'help')")
    if len(args) not in (3, 6):
        raise ValueError(f"'{cmd}' needs 3 or 6 numbers, got {len(args)}")
    values = _parse_numbers(cmd, args)

    pos = values[:3]
    rot = pin.rpy.rpyToMatrix(*np.radians(values[3:])) if len(values) == 6 else None

    if cmd == "goto":
        return pos.copy(), (rot if rot is not None else R_des.copy())
    # move: translate, then rotate about world axes (pre-multiply).
    return x_des + pos, (rot @ R_des if rot is not None else R_des.copy())


def parse_wrench_command(line: str) -> Tuple[np.ndarray, float | None]:
    """Parse a `force` or `pulse` command into a world-frame EE wrench.

    Args:
        line: "force fx fy fz [tx ty tz]", "force off" or
            "pulse fx fy fz [tx ty tz] T".

    Returns:
        Wrench [fx, fy, fz, tx, ty, tz] (6,) in N and Nm, and the pulse
        duration in seconds (None for a held force).

    Raises:
        ValueError: Unknown command or wrong number / type of arguments.
    """
    parts = line.split()
    if not parts:
        raise ValueError("empty command")
    cmd, args = parts[0].lower(), parts[1:]

    if cmd == "force":
        if len(args) == 1 and args[0].lower() == "off":
            return np.zeros(6), None
        if len(args) not in (3, 6):
            raise ValueError(f"'force' needs 3 or 6 numbers or 'off', got {len(args)}")
        values = _parse_numbers(cmd, args)
        return np.concatenate([values, np.zeros(6 - len(values))]), None

    if cmd == "pulse":
        if len(args) not in (4, 7):
            raise ValueError(f"'pulse' needs 3 or 6 numbers plus a duration, got {len(args)}")
        values = _parse_numbers(cmd, args)
        duration = float(values[-1])
        if duration <= 0.0:
            raise ValueError("'pulse' duration must be positive")
        wrench = values[:-1]
        return np.concatenate([wrench, np.zeros(6 - len(wrench))]), duration

    raise ValueError(f"unknown command '{cmd}' (type 'help')")


def predicted_deflection(
    gains: ImpedanceGains, wrench: np.ndarray
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Static EE deflection the spring alone predicts for a constant wrench.

    At equilibrium Jᵀ(K_p·e) + Jᵀ·W = 0, so K_p·e = −W and the EE settles at
    x − x_des = K_pos⁻¹·F and log3(R·R_desᵀ) = K_ori⁻¹·τ (J full row rank).

    Args:
        gains: Controller gains (3x3 or 6x6).
        wrench: External wrench [F(3), τ(3)] in the world frame.

    Returns:
        Predicted Δx (3,) in metres, and Δθ (3,) in radians (None in 3D mode,
        where nothing resists a torque).
    """
    dx = np.linalg.solve(gains.K_p[:3, :3], wrench[:3])
    if gains.K_p.shape[0] != 6:
        return dx, None
    return dx, np.linalg.solve(gains.K_p[3:, 3:], wrench[3:])


# ---------------------------------------------------------------------------
# External wrench on the EE
# ---------------------------------------------------------------------------

def apply_ee_wrench(mj_model, mj_data, site_id: int, wrench: np.ndarray) -> None:
    """Set data.qfrc_applied to a world-frame wrench acting at the EE site.

    The arm is driven through ctrl and mouse pushes through xfrc_applied, so
    qfrc_applied is free for this. It is overwritten every call.
    """
    import mujoco

    mj_data.qfrc_applied[:] = 0.0
    if not np.any(wrench):
        return
    mujoco.mj_applyFT(
        mj_model, mj_data,
        np.asarray(wrench[:3], dtype=float), np.asarray(wrench[3:], dtype=float),
        mj_data.site_xpos[site_id].copy(), int(mj_model.site_bodyid[site_id]),
        mj_data.qfrc_applied,
    )


def run_force_probe(
    wrench: np.ndarray,
    gains: ImpedanceGains,
    q_init: np.ndarray | None = None,
    duration: float = 4.0,
) -> Dict[str, np.ndarray]:
    """Hold the start pose under impedance control while a constant wrench pushes the EE.

    Args:
        wrench: External wrench [F(3), τ(3)] in the world frame (N, Nm).
        gains: Controller gains (3x3 or 6x6).
        q_init: Initial joint config. Defaults to Q_HOME.
        duration: Simulation duration (s).

    Returns:
        Dictionary with keys: time, ee_pos, deflection (EE − target, m),
        ori_deflection (log3(R·R_desᵀ), rad).
    """
    import mujoco

    if q_init is None:
        q_init = Q_HOME.copy()
    is_6d = gains.K_p.shape[0] == 6

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)
    x_des, R_des = get_ee_pose(pin_model, pin_data, ee_fid, q_init)

    mj_data.qpos[:NUM_JOINTS] = q_init
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    n_steps = int(duration / DT)
    time_arr = np.zeros(n_steps)
    ee_pos_arr = np.zeros((n_steps, 3))
    ori_arr = np.zeros((n_steps, 3))

    for i in range(n_steps):
        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()
        tau = compute_impedance_torque(
            pin_model, pin_data, ee_fid, q, qd,
            x_des, R_des if is_6d else None, None, gains,
        )
        apply_arm_torques(mj_model, mj_data, clip_torques(tau))
        apply_ee_wrench(mj_model, mj_data, site_id, wrench)
        mujoco.mj_step(mj_model, mj_data)

        time_arr[i] = mj_data.time
        ee_pos_arr[i] = mj_data.site_xpos[site_id]
        # Orientation from Pinocchio FK: the MuJoCo pinch site shares the EE
        # origin but its axes are turned 90° about z relative to the Pinocchio frame.
        _, R_cur = get_ee_pose(pin_model, pin_data, ee_fid, mj_data.qpos[:NUM_JOINTS])
        ori_arr[i] = -orientation_error(R_des, R_cur)

    return {
        "time": time_arr,
        "ee_pos": ee_pos_arr,
        "deflection": ee_pos_arr - x_des,
        "ori_deflection": ori_arr,
    }


# ---------------------------------------------------------------------------
# Slider panel
# ---------------------------------------------------------------------------

# (label, min, max, resolution)
_GAIN_SLIDERS = (
    ("K_pos  (N/m)", 10.0, 5000.0, 10.0),
    ("D_pos  (N·s/m)", 0.0, 500.0, 1.0),
    ("K_ori  (Nm/rad)", 1.0, 1000.0, 1.0),
    ("D_ori  (Nm·s/rad)", 0.0, 100.0, 0.5),
)
_FORCE_SLIDERS = (
    ("Fx (N)", -50.0, 50.0, 0.5),
    ("Fy (N)", -50.0, 50.0, 0.5),
    ("Fz (N)", -50.0, 50.0, 0.5),
    ("Tx (Nm)", -5.0, 5.0, 0.1),
    ("Ty (Nm)", -5.0, 5.0, 0.1),
    ("Tz (Nm)", -5.0, 5.0, 0.1),
)


def _effective_gains(state: dict) -> Tuple[float, float, float, float]:
    """(K_pos, D_pos, K_ori, D_ori) the controller uses for the current state."""
    if state["custom_gains"] is not None:
        return state["custom_gains"]
    _, k_pos, d_pos = GAIN_PRESETS[state["preset"]]
    return (k_pos, d_pos, k_pos / 10.0, d_pos / 10.0)


def run_panel(state: dict, lock: threading.Lock) -> None:
    """Tk window with sliders for gains, mode and external wrench (blocks; run in a thread).

    Moving a gain slider switches to custom gains. The force sliders act only
    while "Hold force" is ticked; "Pulse" applies them once for the chosen time.
    Gains and mode changed elsewhere (terminal, Home/End keys) are mirrored
    back into the sliders every 250 ms.
    """
    import tkinter as tk

    root = tk.Tk()
    root.title("b1_1 impedance — parameters")
    syncing = False  # True while the panel itself sets slider values

    def on_gain(_value: str = "") -> None:
        if syncing:
            return
        with lock:
            state["custom_gains"] = tuple(v.get() for v in gain_vars)

    def push_wrench(_value: str = "") -> None:
        if syncing or not hold_var.get():
            return
        with lock:
            state["wrench"] = np.array([v.get() for v in force_vars])
            state["pulse_s"], state["wrench_until"] = None, None

    def on_hold() -> None:
        if hold_var.get():
            push_wrench()
        else:
            with lock:
                state["wrench"], state["pulse_s"] = np.zeros(6), None
                state["wrench_until"] = None

    def on_pulse() -> None:
        hold_var.set(False)
        with lock:
            state["wrench"] = np.array([v.get() for v in force_vars])
            state["pulse_s"], state["wrench_until"] = pulse_var.get(), None

    def on_zero_force() -> None:
        nonlocal syncing
        syncing = True
        for v in force_vars:
            v.set(0.0)
        syncing = False
        hold_var.set(False)
        on_hold()

    def on_mode() -> None:
        with lock:
            state["is_6d"] = mode_var.get() == "6d"

    def on_presets() -> None:
        with lock:
            state["custom_gains"] = None

    def on_reset() -> None:
        with lock:
            state["reset"] = True
        hold_var.set(False)

    gains_box = tk.LabelFrame(root, text="Impedance gains", padx=6, pady=4)
    gains_box.pack(fill="x", padx=6, pady=4)
    gain_vars = []
    for label, lo, hi, res in _GAIN_SLIDERS:
        var = tk.DoubleVar()
        tk.Scale(gains_box, label=label, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", length=320, command=on_gain).pack(fill="x")
        gain_vars.append(var)
    row = tk.Frame(gains_box)
    row.pack(fill="x", pady=2)
    mode_var = tk.StringVar(value="6d")
    tk.Radiobutton(row, text="6D", variable=mode_var, value="6d", command=on_mode).pack(side="left")
    tk.Radiobutton(row, text="3D (orientation free)", variable=mode_var, value="3d",
                   command=on_mode).pack(side="left")
    tk.Button(row, text="Use presets", command=on_presets).pack(side="right")

    force_box = tk.LabelFrame(root, text="External wrench on EE (world frame)", padx=6, pady=4)
    force_box.pack(fill="x", padx=6, pady=4)
    force_vars = []
    for label, lo, hi, res in _FORCE_SLIDERS:
        var = tk.DoubleVar(value=0.0)
        tk.Scale(force_box, label=label, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", length=320, command=push_wrench).pack(fill="x")
        force_vars.append(var)
    pulse_var = tk.DoubleVar(value=0.2)
    tk.Scale(force_box, label="Pulse duration (s)", variable=pulse_var, from_=0.05, to=2.0,
             resolution=0.05, orient="horizontal", length=320).pack(fill="x")
    row = tk.Frame(force_box)
    row.pack(fill="x", pady=2)
    hold_var = tk.BooleanVar(value=False)
    tk.Checkbutton(row, text="Hold force", variable=hold_var, command=on_hold).pack(side="left")
    tk.Button(row, text="Pulse", command=on_pulse).pack(side="left", padx=4)
    tk.Button(row, text="Zero", command=on_zero_force).pack(side="left")

    tk.Button(root, text="Reset arm to Q_HOME", command=on_reset).pack(fill="x", padx=6, pady=6)

    def mirror_state() -> None:
        nonlocal syncing
        with lock:
            gains = _effective_gains(state)
            is_6d = state["is_6d"]
        syncing = True
        for var, value in zip(gain_vars, gains):
            if abs(var.get() - value) > 1e-9:
                var.set(value)
        mode_var.set("6d" if is_6d else "3d")
        syncing = False
        root.after(250, mirror_state)

    mirror_state()
    root.mainloop()


# ---------------------------------------------------------------------------
# Interactive viewer mode
# ---------------------------------------------------------------------------

def run_observer(
    q_init: np.ndarray | None = None,
    gains: Tuple[float, float, float, float] | None = None,
    is_6d: bool = True,
    panel: bool = True,
) -> None:
    """Run the b1 controller live in the MuJoCo viewer with terminal commands.

    Markers: green sphere = target, white sphere = EE, capsule = virtual spring
    (green → red with |K·e|), magenta capsule = external force (5 mm/N).
    In 6D mode, faint triad = target orientation, solid triad = current.

    Args:
        q_init: Initial joint config. Defaults to Q_HOME.
        gains: Start gains (K_pos, D_pos, K_ori, D_ori); None = "Medium" preset.
        is_6d: Start in 6D (True) or position-only 3D mode.
        panel: Open the Tk slider panel next to the viewer.
    """
    import mujoco
    import mujoco.viewer

    if q_init is None:
        q_init = Q_HOME.copy()

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)
    x_home, R_home = get_ee_pose(pin_model, pin_data, ee_fid, q_init)

    # Key presses (viewer thread) and typed commands (console thread) only
    # update `state` under `lock`; the simulation loop reads it between steps.
    lock = threading.Lock()
    state = {
        "preset": 1, "is_6d": is_6d, "reset": True,
        "custom_gains": None if gains is None else tuple(gains),
        "x_des": x_home.copy(), "R_des": R_home.copy(),
        "x_cur": x_home.copy(), "R_cur": R_home.copy(),
        "wrench": np.zeros(6), "pulse_s": None, "wrench_until": None,
    }

    def on_key(keycode: int) -> None:
        with lock:
            if keycode in _TARGET_KEYS:
                state["x_des"] = state["x_des"] + _TARGET_KEYS[keycode]
            elif keycode == KEY_HOME:
                state["preset"] = (state["preset"] + 1) % len(GAIN_PRESETS)
                state["custom_gains"] = None
            elif keycode == KEY_END:
                state["is_6d"] = not state["is_6d"]
            elif keycode == KEY_DELETE:
                state["reset"] = True

    def console() -> None:
        while True:
            try:
                line = input("observer> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                continue
            cmd = line.split()[0].lower()
            try:
                if cmd == "help":
                    print(_CONSOLE_HELP)
                elif cmd == "show":
                    with lock:
                        x_d, R_d = state["x_des"], state["R_des"]
                        x_c, R_c = state["x_cur"], state["R_cur"]
                        w = state["wrench"]
                    print(f"  target xyz={np.round(x_d, 4)}  rpy={np.round(rpy_deg(R_d), 2)}")
                    print(f"  EE     xyz={np.round(x_c, 4)}  rpy={np.round(rpy_deg(R_c), 2)}")
                    print(f"  wrench F={np.round(w[:3], 2)} N  tau={np.round(w[3:], 2)} Nm")
                    print(f"  deflection EE-target = {np.round((x_c - x_d) * 1000, 2)} mm")
                elif cmd == "gains":
                    values = _parse_numbers(cmd, line.split()[1:])
                    if len(values) != 4 or values.min() <= 0.0:
                        raise ValueError("'gains' needs 4 positive numbers")
                    with lock:
                        state["custom_gains"] = tuple(values)
                    print(f"  gains K_pos={values[0]:g} D_pos={values[1]:g} "
                          f"K_ori={values[2]:g} D_ori={values[3]:g}")
                elif cmd in ("force", "pulse"):
                    wrench, pulse_s = parse_wrench_command(line)
                    with lock:
                        state["wrench"], state["pulse_s"] = wrench, pulse_s
                        state["wrench_until"] = None
                    print(f"  wrench F={np.round(wrench[:3], 2)} N  tau={np.round(wrench[3:], 2)} Nm"
                          + (f"  for {pulse_s:g} s" if pulse_s else ""))
                else:
                    with lock:
                        x_new, R_new = apply_target_command(
                            line, state["x_des"], state["R_des"], x_home, R_home)
                        state["x_des"], state["R_des"] = x_new, R_new
                    print(f"  target xyz={np.round(x_new, 4)}  rpy={np.round(rpy_deg(R_new), 2)}")
            except ValueError as exc:
                print(f"  error: {exc}")

    sync_every = max(1, int(round((1.0 / 60.0) / DT)))  # ~60 Hz redraw
    print("Impedance observer — close the viewer window to quit.\n" + _HELP_TEXT)
    print(f"\nStart pose: xyz={np.round(x_home, 4)}  rpy={np.round(rpy_deg(R_home), 2)}")
    print(_CONSOLE_HELP)
    threading.Thread(target=console, daemon=True).start()
    if panel:
        threading.Thread(target=run_panel, args=(state, lock), daemon=True).start()

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
            with lock:
                reset = state["reset"]
                if reset:
                    state["x_des"], state["R_des"] = x_home.copy(), R_home.copy()
                    state["wrench"], state["pulse_s"] = np.zeros(6), None
                    state["wrench_until"] = None
                    state["reset"] = False
                # A pulse starts counting at the first step after the command.
                if state["pulse_s"] is not None:
                    if state["wrench_until"] is None:
                        state["wrench_until"] = mj_data.time + state["pulse_s"]
                    elif mj_data.time >= state["wrench_until"]:
                        state["wrench"], state["pulse_s"] = np.zeros(6), None
                        state["wrench_until"] = None
                is_6d = state["is_6d"]
                preset = state["preset"]
                custom = state["custom_gains"]
                x_des, R_des = state["x_des"], state["R_des"]
                wrench = state["wrench"]
            if reset:
                with viewer.lock():
                    mujoco.mj_resetData(mj_model, mj_data)
                    mj_data.qpos[:NUM_JOINTS] = q_init
                    mujoco.mj_forward(mj_model, mj_data)

            gains = preset_gains(preset, is_6d) if custom is None else gains_from_scalars(*custom, is_6d)

            q = mj_data.qpos[:NUM_JOINTS].copy()
            qd = mj_data.qvel[:NUM_JOINTS].copy()
            tau = compute_impedance_torque(
                pin_model, pin_data, ee_fid, q, qd,
                x_des, R_des if is_6d else None, None, gains,
            )
            # compute_impedance_torque leaves pin_data.oMf at the current q.
            x_cur = pin_data.oMf[ee_fid].translation.copy()
            R_cur = pin_data.oMf[ee_fid].rotation.copy()

            tau_clipped = clip_torques(tau)
            saturated |= bool(np.any(tau_clipped != tau))
            apply_arm_torques(mj_model, mj_data, tau_clipped)
            apply_ee_wrench(mj_model, mj_data, site_id, wrench)
            mujoco.mj_step(mj_model, mj_data)
            step += 1

            if step % sync_every:
                continue

            # --- redraw (~60 Hz) ---
            with lock:
                state["x_cur"], state["R_cur"] = x_cur, R_cur
            err = x_des - x_cur
            spring_force = float(np.linalg.norm(gains.K_p[:3, :3] @ err))
            red = min(1.0, spring_force / SPRING_FORCE_FULL_RED)
            ori_err_deg = np.degrees(np.linalg.norm(orientation_error(R_des, R_cur)))
            dx_pred, dth_pred = predicted_deflection(gains, wrench)

            with viewer.lock():
                scn = viewer.user_scn
                scn.ngeom = 0
                _add_sphere(scn, x_des, 0.015, [0.1, 0.9, 0.2, 0.6])
                _add_sphere(scn, x_cur, 0.008, [1.0, 1.0, 1.0, 1.0])
                _add_segment(scn, x_cur, x_des, 0.004, [red, 1.0 - red, 0.1, 0.9])
                if np.any(wrench[:3]):
                    _add_segment(scn, x_cur, x_cur + FORCE_ARROW_SCALE * wrench[:3],
                                 0.006, [1.0, 0.1, 0.9, 0.9])
                if is_6d:
                    _add_axes(scn, x_des, R_des, 0.10, 0.35)
                    _add_axes(scn, x_cur, R_cur, 0.07, 1.0)

            if custom is None:
                label, k_pos, d_pos = GAIN_PRESETS[preset]
                gain_text = f"{label}  K={k_pos:.0f}  D={d_pos:.0f}"
            else:
                gain_text = "Custom  K={:g}/{:g}  D={:g}/{:g}".format(
                    custom[0], custom[2], custom[1], custom[3])
            rpy_des = rpy_deg(R_des)
            dx_meas = -err * 1000
            pred_ori = ("-" if dth_pred is None
                        else f"{np.degrees(np.linalg.norm(dth_pred)):.2f} deg")
            viewer.set_texts([
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "Mode\nStiffness\nTarget xyz (m)\nTarget rpy (deg)\nPosition error\n"
                 "Spring force |K·e|\nOrientation error\nTorque limit\n"
                 "External F (N)\nExternal tau (Nm)\nDeflection (mm)\nPredicted F/K (mm)\n"
                 "Predicted tau/K",
                 f"{'6D' if is_6d else '3D (orientation free)'}\n"
                 f"{gain_text}\n"
                 f"{x_des[0]:.3f} {x_des[1]:.3f} {x_des[2]:.3f}\n"
                 f"{rpy_des[0]:.1f} {rpy_des[1]:.1f} {rpy_des[2]:.1f}\n"
                 f"{np.linalg.norm(err) * 1000:.1f} mm\n"
                 f"{spring_force:.1f} N\n"
                 f"{ori_err_deg:.1f} deg\n"
                 f"{'SATURATED' if saturated else 'ok'}\n"
                 f"{wrench[0]:.1f} {wrench[1]:.1f} {wrench[2]:.1f}\n"
                 f"{wrench[3]:.1f} {wrench[4]:.1f} {wrench[5]:.1f}\n"
                 f"{dx_meas[0]:.1f} {dx_meas[1]:.1f} {dx_meas[2]:.1f}\n"
                 f"{dx_pred[0] * 1000:.1f} {dx_pred[1] * 1000:.1f} {dx_pred[2] * 1000:.1f}\n"
                 f"{pred_ori}"),
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the viewer, or run a headless force probe with --probe."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", nargs="+", type=float, metavar="F",
                        help="headless: hold wrench fx fy fz [tx ty tz] at Q_HOME and "
                             "compare the deflection with K⁻¹·W")
    parser.add_argument("--gains", nargs=4, type=float, default=None,
                        metavar=("K_POS", "D_POS", "K_ORI", "D_ORI"),
                        help="start gains (viewer default: Medium preset; "
                             "--probe default: 500 50 50 5, the b1 6D demo)")
    parser.add_argument("--mode", choices=("6d", "3d"), default="6d",
                        help="start in 6D (position + orientation) or 3D (position only)")
    parser.add_argument("--no-panel", action="store_true", help="do not open the slider panel")
    parser.add_argument("--duration", type=float, default=4.0, help="--probe duration (s)")
    args = parser.parse_args()

    if args.probe is None:
        run_observer(gains=args.gains, is_6d=args.mode == "6d", panel=not args.no_panel)
        return

    if len(args.probe) not in (3, 6):
        parser.error("--probe needs 3 or 6 numbers")
    wrench = np.concatenate([args.probe, np.zeros(6 - len(args.probe))])
    gain_values = args.gains if args.gains is not None else [500.0, 50.0, 50.0, 5.0]
    gains = gains_from_scalars(*gain_values, is_6d=args.mode == "6d")
    result = run_force_probe(wrench, gains, duration=args.duration)
    dx_pred, dth_pred = predicted_deflection(gains, wrench)

    print(f"Wrench F={wrench[:3]} N  tau={wrench[3:]} Nm   gains={gain_values}  mode={args.mode}")
    print(f"  Deflection after {args.duration:g} s: {result['deflection'][-1] * 1000} mm")
    print(f"  Predicted K_pos⁻¹·F:            {dx_pred * 1000} mm")
    print(f"  Rotation after {args.duration:g} s:   "
          f"{np.degrees(result['ori_deflection'][-1])} deg")
    print(f"  Predicted K_ori⁻¹·tau:          {np.degrees(dth_pred)} deg")


if __name__ == "__main__":
    main()
