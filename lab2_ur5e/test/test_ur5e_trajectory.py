"""Tests for UR5e trajectory generation (b1 module)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

LAB2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(LAB2_SRC) not in sys.path:
    sys.path.insert(0, str(LAB2_SRC))

from ur5e_common import Q_HOME, NUM_JOINTS, VELOCITY_LIMITS
from b1_trajectory_generation import (
    cubic_trajectory,
    quintic_trajectory,
    trapezoidal_trajectory,
    multi_segment_trajectory,
)


Q_END = np.array([0.0, -np.pi / 2, np.pi / 3, -np.pi / 3, -np.pi / 2, 0.0])


class TestCubicTrajectory(unittest.TestCase):

    def test_endpoint_positions(self) -> None:
        traj = cubic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        np.testing.assert_allclose(traj[0].q, Q_HOME, atol=1e-10)
        np.testing.assert_allclose(traj[-1].q, Q_END, atol=1e-6)

    def test_zero_endpoint_velocity(self) -> None:
        traj = cubic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertLess(np.linalg.norm(traj[0].qd), 1e-10)
        self.assertLess(np.linalg.norm(traj[-1].qd), 1e-6)


class TestQuinticTrajectory(unittest.TestCase):

    def test_zero_endpoint_velocity(self) -> None:
        traj = quintic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertLess(np.linalg.norm(traj[0].qd), 1e-10)
        self.assertLess(np.linalg.norm(traj[-1].qd), 1e-6)

    def test_zero_endpoint_acceleration(self) -> None:
        traj = quintic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertLess(np.linalg.norm(traj[0].qdd), 1e-10)
        self.assertLess(np.linalg.norm(traj[-1].qdd), 1e-6)


class TestTrapezoidalTrajectory(unittest.TestCase):

    def test_respects_velocity_limits(self) -> None:
        v_max = VELOCITY_LIMITS * 0.5
        a_max = np.full(NUM_JOINTS, 4.0)
        traj = trapezoidal_trajectory(Q_HOME, Q_END, v_max, a_max, 0.01)
        for pt in traj:
            max_vel = np.max(np.abs(pt.qd))
            self.assertLess(max_vel, np.max(v_max) + 0.01)

    def test_endpoints(self) -> None:
        v_max = VELOCITY_LIMITS * 0.5
        a_max = np.full(NUM_JOINTS, 4.0)
        traj = trapezoidal_trajectory(Q_HOME, Q_END, v_max, a_max, 0.01)
        np.testing.assert_allclose(traj[0].q, Q_HOME, atol=1e-6)
        np.testing.assert_allclose(traj[-1].q, Q_END, atol=1e-6)

    def test_synchronized_joints_reach_goal(self) -> None:
        """Non-limiting joints must still land on the goal (regression).

        Several joints hit v_max here; the old synchronization reused v_max
        with a_max for them and overshot/undershot by ~11 deg.
        """
        q0 = np.deg2rad([0.0, -90.0, 60.0, -60.0, -90.0, 0.0])
        q1 = np.deg2rad([-177.7, -101.5, 107.2, -95.8, -90.0, -177.7])
        v_max = VELOCITY_LIMITS * 0.5
        a_max = np.full(NUM_JOINTS, 4.0)
        traj = trapezoidal_trajectory(q0, q1, v_max, a_max, 0.002)
        np.testing.assert_allclose(traj[-1].q, q1, atol=1e-6)
        q = np.array([pt.q for pt in traj])
        qd = np.array([pt.qd for pt in traj])
        self.assertTrue(np.all(np.abs(qd) <= v_max + 1e-9))
        # Position must be the integral of velocity (no hidden jumps).
        np.testing.assert_allclose(np.diff(q, axis=0) / 0.002,
                                   0.5 * (qd[1:] + qd[:-1]), atol=0.05)


class TestMultiSegmentTrajectory(unittest.TestCase):

    def test_lands_on_last_waypoint(self) -> None:
        q_mid = np.array([-np.pi / 4, -np.pi / 2, np.pi / 4,
                           -np.pi / 4, -np.pi / 2, 0.0])
        waypoints = [Q_HOME, q_mid, Q_END]
        traj = multi_segment_trajectory(waypoints, [1.5, 1.5], 0.01)
        np.testing.assert_allclose(traj[-1].q, Q_END, atol=1e-6)

    def test_point_count(self) -> None:
        traj = quintic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertGreater(len(traj), 100)


if __name__ == "__main__":
    unittest.main()
