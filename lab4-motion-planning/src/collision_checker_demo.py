"""Lab 4 — MuJoCo-based collision checker for the UR5e + Robotiq 2F-85.

Generated purely from ``tasks/specs/collision_checker.md`` (spec-sufficiency
experiment); the existing implementation was not consulted.

Collision truth comes exclusively from the MuJoCo scene returned by
``lab4_common.load_mujoco_model``. Pinocchio is loaded only so callers can use
``cc.model`` / ``cc.data`` / ``cc.ee_fid`` for forward kinematics.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from lab4_common import (
    OBSTACLES,
    URDF_PATH,
    ObstacleSpec,
    load_mujoco_model,
    load_pinocchio_model,
)

_GRIPPER_PREFIX = "2f85_"
_TABLE_NAME = "table"
_ZERO_TOL = 1e-8
_DIST_MAX = 10.0

Pair = tuple[int, int]


class CollisionChecker:
    """Configuration / segment collision checks and signed distances.

    Monitored pairs are robot-vs-obstacle, robot-vs-table and (optionally)
    robot self-collision pairs filtered by kinematic-tree adjacency. Gripper
    internal pairs and the floor plane are never monitored; visual geoms
    (``contype == 0``) are ignored. The instance owns its own ``MjData``.
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        obstacle_specs: list[ObstacleSpec] | tuple[ObstacleSpec, ...] | None = None,
        self_collision: bool = True,
        adjacency_gap: int = 1,
        include_table: bool | None = None,
    ) -> None:
        """Build the MuJoCo scene, Pinocchio model and monitored pair buckets.

        Args:
            urdf_path: URDF for Pinocchio (defaults to ``URDF_PATH``).
            obstacle_specs: Obstacle boxes. ``None`` means the default
                ``OBSTACLES`` plus the table; an empty sequence means no
                environment at all unless ``include_table`` is given.
            self_collision: Whether to monitor robot self-collision pairs.
            adjacency_gap: Self pairs with kinematic tree distance
                ``<= adjacency_gap`` are skipped.
            include_table: Whether the table is in the scene (``None`` = auto).
        """
        if obstacle_specs is None:
            resolved = list(OBSTACLES)
            table = True if include_table is None else include_table
        else:
            resolved = list(obstacle_specs)
            table = (len(resolved) > 0) if include_table is None else include_table

        self.obstacle_specs: list[ObstacleSpec] = resolved
        self.self_collision = self_collision
        self.adjacency_gap = adjacency_gap
        self._include_table: bool = bool(table)

        self.mj_model, self.mj_data = load_mujoco_model(
            obstacle_specs=tuple(resolved), include_table=self._include_table
        )
        self.model, self.data, self.ee_fid = load_pinocchio_model(urdf_path or URDF_PATH)

        self._classify_geoms()
        self._build_pairs()

    # ------------------------------------------------------------------ setup
    def _body_name(self, body_id: int) -> str:
        """Return the MuJoCo body name (empty string if unnamed)."""
        name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name or ""

    def _classify_geoms(self) -> None:
        """Split collision geoms into robot, table and obstacle geoms."""
        m = self.mj_model
        env_names = {spec.name for spec in self.obstacle_specs}
        if self._include_table:
            env_names.add(_TABLE_NAME)

        env_bodies: set[int] = set()
        robot_bodies: set[int] = set()
        for b in range(m.nbody):
            name = self._body_name(b)
            if name in env_names:
                env_bodies.add(b)
            elif b != 0 and name != "world":
                robot_bodies.add(b)

        robot_geoms: list[int] = []
        env_geoms: list[int] = []
        table_geoms: set[int] = set()
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0:
                continue
            b = int(m.geom_bodyid[g])
            if b in robot_bodies:
                robot_geoms.append(g)
            elif b in env_bodies:
                env_geoms.append(g)
                if self._body_name(b) == _TABLE_NAME:
                    table_geoms.add(g)

        self._robot_geoms: list[int] = robot_geoms
        self._n_robot_geoms: int = len(robot_geoms)
        self._obstacle_ids: list[int] = env_geoms
        self._table_ids: list[int] = [g for g in env_geoms if g in table_geoms]

    def _tree_distance(self, b1: int, b2: int) -> int:
        """Number of edges between two bodies in the MuJoCo kinematic tree."""
        parent = self.mj_model.body_parentid
        depth: dict[int, int] = {}
        cur, d = b1, 0
        while True:
            depth[cur] = d
            if cur == 0:
                break
            cur = int(parent[cur])
            d += 1
        cur, steps = b2, 0
        while cur not in depth:
            cur = int(parent[cur])
            steps += 1
        return depth[cur] + steps

    def _self_pair_allowed(self, a: int, b: int) -> bool:
        """Apply the self-pair rule (spec 4.4) to robot geoms ``a`` and ``b``."""
        ba = int(self.mj_model.geom_bodyid[a])
        bb = int(self.mj_model.geom_bodyid[b])
        if ba == bb:
            return False
        if self._body_name(ba).startswith(_GRIPPER_PREFIX) and self._body_name(
            bb
        ).startswith(_GRIPPER_PREFIX):
            return False
        return self._tree_distance(ba, bb) > self.adjacency_gap

    def _build_pairs(self) -> None:
        """Build the obstacle, table and self pair buckets (spec 4.3)."""
        self._pair_set: set[Pair] = set()
        self._obstacle_pairs: list[Pair] = []
        self._table_pairs: list[Pair] = []
        self._self_pairs: list[Pair] = []
        table = set(self._table_ids)

        def add(pair: Pair, bucket: list[Pair]) -> None:
            if pair not in self._pair_set:
                self._pair_set.add(pair)
                bucket.append(pair)

        for r in self._robot_geoms:
            for e in self._obstacle_ids:
                pair = (min(r, e), max(r, e))
                add(pair, self._table_pairs if e in table else self._obstacle_pairs)

        if self.self_collision:
            rg = self._robot_geoms
            for i in range(len(rg)):
                for j in range(i + 1, len(rg)):
                    a, b = rg[i], rg[j]
                    if self._self_pair_allowed(a, b):
                        add((min(a, b), max(a, b)), self._self_pairs)

    # ---------------------------------------------------------------- queries
    def _set_configuration(self, q: np.ndarray) -> None:
        """Reset state, write arm joints ``q`` and run ``mj_forward``."""
        q = np.asarray(q, dtype=float)
        m, d = self.mj_model, self.mj_data
        d.qpos[: m.nq] = 0.0
        d.qvel[: m.nv] = 0.0
        d.qpos[: len(q)] = q
        mujoco.mj_forward(m, d)

    @property
    def num_collision_pairs(self) -> int:
        """Number of distinct monitored geom pairs."""
        return len(self._pair_set)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Return True if no monitored pair is in contact at ``q``."""
        self._set_configuration(q)
        d = self.mj_data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (min(g1, g2), max(g1, g2)) in self._pair_set:
                return False
        return True

    def is_path_free(
        self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.05
    ) -> bool:
        """Check the straight joint-space segment ``q1 -> q2`` by sampling.

        Args:
            q1: Start configuration (6,).
            q2: End configuration (6,).
            resolution: Max joint-space L2 step between samples [rad].
        """
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        diff = q2 - q1
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return self.is_collision_free(q1)
        n = max(2, math.ceil(dist / resolution) + 1)
        for i in range(n):
            if not self.is_collision_free(q1 + (i / (n - 1)) * diff):
                return False
        return True

    def compute_min_distance(
        self,
        q: np.ndarray,
        include_self: bool = True,
        include_table: bool = True,
        include_obstacles: bool = True,
    ) -> float:
        """Minimum signed distance [m] over the selected monitored pairs.

        Positive = clearance, negative = penetration, 0.0 = touching,
        ``inf`` if no pairs are selected.
        """
        pairs: list[Pair] = []
        if include_obstacles:
            pairs.extend(self._obstacle_pairs)
        if include_table:
            pairs.extend(self._table_pairs)
        if include_self:
            pairs.extend(self._self_pairs)
        self._set_configuration(q)
        if not pairs:
            return float("inf")
        fromto = np.zeros(6)
        best = math.inf
        for g1, g2 in pairs:
            dist = mujoco.mj_geomDistance(
                self.mj_model, self.mj_data, g1, g2, _DIST_MAX, fromto
            )
            if dist < best:
                best = dist
        if abs(best) < _ZERO_TOL:
            return 0.0
        return float(best)

    def compute_min_environment_distance(
        self, q: np.ndarray, include_table: bool = True
    ) -> float:
        """Minimum robot-vs-environment distance (obstacles, optionally table)."""
        return self.compute_min_distance(
            q, include_self=False, include_table=include_table, include_obstacles=True
        )

    def compute_min_obstacle_distance(self, q: np.ndarray) -> float:
        """Minimum robot-vs-obstacle-box distance (table excluded)."""
        return self.compute_min_distance(
            q, include_self=False, include_table=False, include_obstacles=True
        )
