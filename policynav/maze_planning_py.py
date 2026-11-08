"""Python ctypes bindings for libmaze_planning_core.so (GroundTruthMap, RRT*, PathTransformer, STVL Costmap, 2048-rollout MPPI)."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_MAP = DEPLOY_ROOT / "config" / "course_groundtruth.bin"


def _find_lib() -> Path:
    candidates = [
        DEPLOY_ROOT / "build" / "libmaze_planning_core.so",
        DEPLOY_ROOT / "build" / "pkg_maze_planning" / "libmaze_planning_core.so",
        DEPLOY_ROOT / "install" / "pkg_maze_planning" / "lib" / "libmaze_planning_core.so",
        DEPLOY_ROOT / "build" / "isam2_depth" / "libmaze_planning_core.so",
        DEPLOY_ROOT / "install" / "isam2_depth" / "lib" / "libmaze_planning_core.so",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"libmaze_planning_core.so not found in {[str(c) for c in candidates]}. Build src/ first."
    )


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("groundtruth_map_path", ctypes.c_char_p),
        ("prefer_window", ctypes.c_int),
        ("include_return_leg", ctypes.c_int),
        ("mppi_samples", ctypes.c_int),
        ("mppi_horizon_steps", ctypes.c_int),
        ("cruise_speed_m_s", ctypes.c_double),
    ]


class _CState(ctypes.Structure):
    _fields_ = [
        ("timestamp_s", ctypes.c_double),
        ("pos_odom", ctypes.c_double * 3),
        ("vel_odom", ctypes.c_double * 3),
        ("quat_wxyz", ctypes.c_double * 4),
    ]


class _COutput(ctypes.Structure):
    _fields_ = [
        ("pos_ned", ctypes.c_double * 3),
        ("vel_ned", ctypes.c_double * 3),
        ("acc_ned", ctypes.c_double * 3),
        ("yaw_rad", ctypes.c_double),
        ("yawspeed_rad_s", ctypes.c_double),
        ("body_action", ctypes.c_double * 4),
        ("rl_lookaheads_body", ctypes.c_float * 12),
        ("obstacle_hold", ctypes.c_int),
        ("replanned", ctypes.c_int),
        ("solve_time_ms", ctypes.c_double),
        ("min_rollout_cost", ctypes.c_double),
        ("remaining_dist_m", ctypes.c_double),
    ]


def _load_cdll() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_lib()))
    lib.maze_pipeline_create.argtypes = [ctypes.POINTER(_CConfig)]
    lib.maze_pipeline_create.restype = ctypes.c_void_p
    lib.maze_pipeline_destroy.argtypes = [ctypes.c_void_p]
    lib.maze_pipeline_destroy.restype = None
    lib.maze_pipeline_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.maze_pipeline_reset.restype = None
    lib.maze_pipeline_query_groundtruth.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.maze_pipeline_query_groundtruth.restype = ctypes.c_double
    lib.maze_pipeline_query_costmap.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.maze_pipeline_query_costmap.restype = ctypes.c_double
    lib.maze_pipeline_inject_pillar.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.maze_pipeline_inject_pillar.restype = None
    lib.maze_pipeline_clear_dynamic_obstacles.argtypes = [ctypes.c_void_p]
    lib.maze_pipeline_clear_dynamic_obstacles.restype = None
    lib.maze_pipeline_set_map_to_odom.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.maze_pipeline_set_map_to_odom.restype = None
    lib.maze_pipeline_plan_point_to_point.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.maze_pipeline_plan_point_to_point.restype = ctypes.c_int
    lib.maze_pipeline_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CState),
        ctypes.POINTER(_COutput),
    ]
    lib.maze_pipeline_step.restype = None
    return lib


class MazePlanningPipeline:
    """Python interface to the C++17 GroundTruthMap + Informed RRT* + PathTransformer + STVL Costmap + 2048-rollout MPPI."""

    def __init__(
        self,
        groundtruth_map_path: str | os.PathLike | None = None,
        prefer_window: bool = True,
        include_return_leg: bool = False,
        mppi_samples: int = 2048,
        mppi_horizon_steps: int = 16,
        cruise_speed_m_s: float = 2.65,
    ):
        self._lib = _load_cdll()
        gt_bytes = str(groundtruth_map_path or DEFAULT_GT_MAP).encode("utf-8")
        cfg = _CConfig(
            groundtruth_map_path=gt_bytes,
            prefer_window=1 if prefer_window else 0,
            include_return_leg=1 if include_return_leg else 0,
            mppi_samples=int(mppi_samples),
            mppi_horizon_steps=int(mppi_horizon_steps),
            cruise_speed_m_s=float(cruise_speed_m_s),
        )
        self._h = self._lib.maze_pipeline_create(ctypes.byref(cfg))
        self._t = 0.0

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.maze_pipeline_destroy(self._h)
            self._h = None

    def reset(self, start_pos_map: tuple[float, float, float] = (0.9144, -2.5146, -0.78)) -> None:
        p = (ctypes.c_double * 3)(*map(float, start_pos_map))
        self._lib.maze_pipeline_reset(self._h, p)
        self._t = 0.0

    def query_groundtruth(self, pos_map: list[float] | np.ndarray) -> tuple[float, np.ndarray]:
        p = (ctypes.c_double * 3)(float(pos_map[0]), float(pos_map[1]), float(pos_map[2]))
        g = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        d = self._lib.maze_pipeline_query_groundtruth(self._h, p, g)
        return float(d), np.array([g[0], g[1], g[2]], dtype=np.float64)

    def query_costmap(self, pos_map: list[float] | np.ndarray) -> float:
        p = (ctypes.c_double * 3)(float(pos_map[0]), float(pos_map[1]), float(pos_map[2]))
        return float(self._lib.maze_pipeline_query_costmap(self._h, p))

    def inject_pillar(
        self, x_map: float, y_map: float, radius_m: float = 0.10, height_m: float = 1.65
    ) -> None:
        self._lib.maze_pipeline_inject_pillar(
            self._h, float(x_map), float(y_map), float(radius_m), float(height_m)
        )

    def clear_dynamic_obstacles(self) -> None:
        self._lib.maze_pipeline_clear_dynamic_obstacles(self._h)

    def set_map_to_odom(
        self,
        t_xyz: tuple[float, float, float] | np.ndarray,
        q_wxyz: tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        t = (ctypes.c_double * 3)(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2]))
        q = (ctypes.c_double * 4)(float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]))
        self._lib.maze_pipeline_set_map_to_odom(self._h, t, q)

    def plan_point_to_point(
        self, start_map: list[float] | np.ndarray, goal_map: list[float] | np.ndarray
    ) -> np.ndarray:
        s = (ctypes.c_double * 3)(float(start_map[0]), float(start_map[1]), float(start_map[2]))
        g = (ctypes.c_double * 3)(float(goal_map[0]), float(goal_map[1]), float(goal_map[2]))
        max_wps = 256
        buf = (ctypes.c_double * (3 * max_wps))()
        n = self._lib.maze_pipeline_plan_point_to_point(self._h, s, g, buf, max_wps)
        n_clamped = min(int(n), max_wps)
        return np.frombuffer(buf, dtype=np.float64, count=3 * n_clamped).reshape(n_clamped, 3).copy()

    def step(
        self,
        pos_odom: np.ndarray | list[float],
        vel_odom: np.ndarray | list[float],
        quat_wxyz: np.ndarray | list[float] = (1.0, 0.0, 0.0, 0.0),
        dt_s: float = 0.02,
    ) -> dict:
        self._t += dt_s
        st = _CState()
        st.timestamp_s = float(self._t)
        st.pos_odom[:] = [float(pos_odom[0]), float(pos_odom[1]), float(pos_odom[2])]
        st.vel_odom[:] = [float(vel_odom[0]), float(vel_odom[1]), float(vel_odom[2])]
        st.quat_wxyz[:] = [
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        ]
        out = _COutput()
        self._lib.maze_pipeline_step(self._h, ctypes.byref(st), ctypes.byref(out))
        return {
            "pos_ned": np.array([out.pos_ned[0], out.pos_ned[1], out.pos_ned[2]], dtype=np.float32),
            "vel_ned": np.array([out.vel_ned[0], out.vel_ned[1], out.vel_ned[2]], dtype=np.float32),
            "acc_ned": np.array([out.acc_ned[0], out.acc_ned[1], out.acc_ned[2]], dtype=np.float32),
            "yaw_rad": float(out.yaw_rad),
            "yawspeed_rad_s": float(out.yawspeed_rad_s),
            "body_action": np.array(
                [out.body_action[0], out.body_action[1], out.body_action[2], out.body_action[3]],
                dtype=np.float32,
            ),
            "rl_lookaheads_body": np.array(list(out.rl_lookaheads_body), dtype=np.float32).reshape(4, 3),
            "obstacle_hold": bool(out.obstacle_hold),
            "replanned": bool(out.replanned),
            "solve_time_ms": float(out.solve_time_ms),
            "min_rollout_cost": float(out.min_rollout_cost),
            "remaining_dist_m": float(out.remaining_dist_m),
        }
