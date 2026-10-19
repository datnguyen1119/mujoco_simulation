"""Tests for the Lab 3 impedance observer (b1_1): command parsing and F = K·Δx."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from b1_1_impedance_observer import (
    apply_target_command,
    gains_from_scalars,
    parse_wrench_command,
    predicted_deflection,
    run_force_probe,
)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestTargetCommand(unittest.TestCase):
    """Tests for goto / move / home."""

    def setUp(self) -> None:
        self.x_home = np.array([-0.134, 0.492, 0.332])
        self.R_home = np.diag([1.0, -1.0, -1.0])  # rpy = (180, 0, 0) deg
        self.x_des = self.x_home + np.array([0.0, 0.0, 0.05])
        self.R_des = rot_z(0.3) @ self.R_home

    def _run(self, line: str):
        return apply_target_command(line, self.x_des, self.R_des, self.x_home, self.R_home)

    def test_goto_position_keeps_orientation(self) -> None:
        x, R = self._run("goto 0.1 0.4 0.5")
        self.assertTrue(np.allclose(x, [0.1, 0.4, 0.5], atol=1e-12))
        self.assertTrue(np.allclose(R, self.R_des, atol=1e-12))

    def test_goto_rpy_matches_home(self) -> None:
        _, R = self._run("goto 0 0 0 180 0 0")
        self.assertTrue(np.allclose(R, self.R_home, atol=1e-9))

    def test_move_rotates_about_world_axes(self) -> None:
        x, R = self._run("move 0.01 0 -0.02 0 0 30")
        self.assertTrue(np.allclose(x, self.x_des + [0.01, 0.0, -0.02], atol=1e-12))
        self.assertTrue(np.allclose(R, rot_z(np.radians(30.0)) @ self.R_des, atol=1e-9))

    def test_home(self) -> None:
        x, R = self._run("home")
        self.assertTrue(np.allclose(x, self.x_home, atol=1e-12))
        self.assertTrue(np.allclose(R, self.R_home, atol=1e-12))

    def test_bad_input_raises(self) -> None:
        for line in ("", "fly 1 2 3", "goto 1 2", "move a b c", "goto 1 2 3 4", "home 1"):
            with self.assertRaises(ValueError, msg=line):
                self._run(line)


class TestWrenchCommand(unittest.TestCase):
    """Tests for force / pulse."""

    def test_force_pads_torque(self) -> None:
        w, T = parse_wrench_command("force 0 0 -20")
        self.assertTrue(np.allclose(w, [0, 0, -20, 0, 0, 0], atol=1e-12))
        self.assertIsNone(T)

    def test_force_off(self) -> None:
        w, T = parse_wrench_command("force off")
        self.assertTrue(np.allclose(w, np.zeros(6), atol=1e-12))
        self.assertIsNone(T)

    def test_pulse_with_torque(self) -> None:
        w, T = parse_wrench_command("pulse 10 0 0 0 0 2 0.5")
        self.assertTrue(np.allclose(w, [10, 0, 0, 0, 0, 2], atol=1e-12))
        self.assertAlmostEqual(T, 0.5)

    def test_bad_input_raises(self) -> None:
        for line in ("force 1 2", "force x y z", "pulse 1 2 3", "pulse 1 2 3 0", "push 1 2 3"):
            with self.assertRaises(ValueError, msg=line):
                parse_wrench_command(line)


class TestForceProbe(unittest.TestCase):
    """A held wrench should deflect the EE by K⁻¹·W (spring law of the controller)."""

    def test_force_deflection_matches_spring(self) -> None:
        gains = gains_from_scalars(500.0, 100.0, 50.0, 10.0, is_6d=True)
        wrench = np.array([0.0, 10.0, -10.0, 0.0, 0.0, 0.0])
        result = run_force_probe(wrench, gains, duration=4.0)
        dx_pred, _ = predicted_deflection(gains, wrench)  # 20 mm along +y and −z
        self.assertTrue(np.allclose(result["deflection"][-1], dx_pred, atol=1e-3),
                        f"{result['deflection'][-1]} vs {dx_pred}")

    def test_torque_rotation_matches_spring(self) -> None:
        gains = gains_from_scalars(500.0, 100.0, 50.0, 10.0, is_6d=True)
        wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        result = run_force_probe(wrench, gains, duration=4.0)
        _, dth_pred = predicted_deflection(gains, wrench)  # 0.04 rad about z
        self.assertTrue(np.allclose(result["ori_deflection"][-1], dth_pred, atol=2e-3),
                        f"{result['ori_deflection'][-1]} vs {dth_pred}")


if __name__ == "__main__":
    unittest.main()
