"""Python ctypes wrapper around libego_planner_core.so for closed-loop sim & testing."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR_BIN = DEPLOY_ROOT / "assets" / "arena_prior.bin"


def _find_library() -> Path:
    candidates = [
        DEPLOY_ROOT / "ego_planner" / "build" / "libego_planner_core.so",
        DEPLOY_ROOT / "build" / "ego_planner" / "libego_planner_core.so",
        DEPLOY_ROOT / "install" / "ego_planner" / "lib" / "libego_planner_core.so",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"libego_planner_core.so not found in {[str(c) for c in candidates]}. Build ego_planner first."
    )


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("arm_authorized", ctypes.c_int),
        ("enable_return_leg", ctypes.c_int),
        ("prefer_window", ctypes.c_int),
        ("cruise_alt_m", ctypes.c_double),
        ("desired_speed_ms", ctypes.c_double),
        ("replan_period_s", ctypes.c_double),
        ("prior_bin_path", ctypes.c_char_p),
    ]


class _CState(ctypes.Structure):
    _fields_ = [
        ("pos_ned", ctypes.c_double * 3),
        ("vel_ned", ctypes.c_double * 3),
        ("quat_wxyz", ctypes.c_double * 4),
        ("rangefinder_m", ctypes.c_double),
        ("armed", ctypes.c_int),
        ("offboard", ctypes.c_int),
        ("landed", ctypes.c_int),
        ("pose_valid", ctypes.c_int),
    ]


class _COutput(ctypes.Structure):
    _fields_ = [
        ("publish_offboard", ctypes.c_int),
        ("publish_arm_cmd", ctypes.c_int),
        ("publish_offboard_switch", ctypes.c_int),
        ("publish_land_cmd", ctypes.c_int),
        ("vel_ned", ctypes.c_double * 3),
        ("yawspeed_rad_s", ctypes.c_double),
        ("body_action", ctypes.c_double * 4),
        ("next_checkpoint", ctypes.c_int),
        ("phase", ctypes.c_char * 32),
        ("why", ctypes.c_char * 64),
        ("result", ctypes.c_char * 64),
        ("active_leg", ctypes.c_char * 32),
        ("active_primitive", ctypes.c_char * 32),
    ]


def _load_cdll() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_library()))
    lib.ego_planner_create.argtypes = [ctypes.POINTER(_CConfig)]
    lib.ego_planner_create.restype = ctypes.c_void_p
    lib.ego_planner_destroy.argtypes = [ctypes.c_void_p]
    lib.ego_planner_destroy.restype = None
    lib.ego_planner_reset.argtypes = [ctypes.c_void_p]
    lib.ego_planner_reset.restype = None
    lib.ego_planner_on_estimator_health.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    lib.ego_planner_on_estimator_health.restype = None
    lib.ego_planner_on_depth_timestamp.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.ego_planner_on_depth_timestamp.restype = None
    lib.ego_planner_on_depth_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.ego_planner_on_depth_frame.restype = None
    lib.ego_planner_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(_CState),
        ctypes.POINTER(_COutput),
    ]
    lib.ego_planner_step.restype = None
    lib.ego_planner_query_esdf.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ego_planner_query_esdf.restype = ctypes.c_double
    return lib


class EgoPlannerV2:
    """Python interface to the C++17 EGO-Planner-V2 MissionManager & ESDF GridMap."""

    def __init__(
        self,
        arm_authorized: bool = True,
        enable_return_leg: bool = False,
        prefer_window: bool = True,
        cruise_alt_m: float = 0.78,
        desired_speed_ms: float = 3.35,
        replan_period_s: float = 0.20,
        prior_bin_path: str | os.PathLike | None = None,
    ):
        self._lib = _load_cdll()
        p_path = str(prior_bin_path or DEFAULT_PRIOR_BIN).encode("utf-8")
        cfg = _CConfig(
            arm_authorized=1 if arm_authorized else 0,
            enable_return_leg=1 if enable_return_leg else 0,
            prefer_window=1 if prefer_window else 0,
            cruise_alt_m=float(cruise_alt_m),
            desired_speed_ms=float(desired_speed_ms),
            replan_period_s=float(replan_period_s),
            prior_bin_path=p_path,
        )
        self._h = self._lib.ego_planner_create(ctypes.byref(cfg))
        self._sim_t = 0.0
        self._ReadyForSimFlight = False

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.ego_planner_destroy(self._h)
            self._h = None

    def reset(self, ready_for_sim_flight: bool = True) -> None:
        self._lib.ego_planner_reset(self._h)
        self._sim_t = 0.0
        self._ReadyForSimFlight = False
        if ready_for_sim_flight:
            # Fast-forward pre-flight WAIT -> ARM -> FLY transition for immediate CourseEnv episodes
            dummy_state = _CState()
            dummy_state.pos_ned[:] = [0.9144, -2.5146, -0.08]
            dummy_state.vel_ned[:] = [0.0, 0.0, 0.0]
            dummy_state.quat_wxyz[:] = [1.0, 0.0, 0.0, 0.0]
            dummy_state.rangefinder_m = 0.08
            dummy_state.armed = 0
            dummy_state.offboard = 0
            dummy_state.landed = 1
            dummy_state.pose_valid = 1
            out = _COutput()
            self._lib.ego_planner_on_estimator_health(self._h, 0.0, 1)
            self._lib.ego_planner_on_depth_timestamp(self._h, 0.0)
            self._lib.ego_planner_step(self._h, 0.0, ctypes.byref(dummy_state), ctypes.byref(out))
            self._lib.ego_planner_on_estimator_health(self._h, 1.1, 1)
            self._lib.ego_planner_on_depth_timestamp(self._h, 1.1)
            self._lib.ego_planner_step(self._h, 1.1, ctypes.byref(dummy_state), ctypes.byref(out))
            dummy_state.armed = 1
            dummy_state.offboard = 1
            dummy_state.landed = 0
            self._lib.ego_planner_on_estimator_health(self._h, 1.2, 1)
            self._lib.ego_planner_on_depth_timestamp(self._h, 1.2)
            self._lib.ego_planner_step(self._h, 1.2, ctypes.byref(dummy_state), ctypes.byref(out))
            self._sim_t = 1.2
            self._ReadyForSimFlight = True

    def query_esdf(self, pos_ned: np.ndarray | list[float]) -> tuple[float, np.ndarray]:
        p = (ctypes.c_double * 3)(float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]))
        g = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        d = self._lib.ego_planner_query_esdf(self._h, p, g)
        return float(d), np.array([g[0], g[1], g[2]], dtype=np.float64)

    def step_raw(
        self,
        t_s: float,
        pos_ned: np.ndarray | tuple[float, float, float],
        vel_ned: np.ndarray | tuple[float, float, float],
        quat_wxyz: np.ndarray | tuple[float, float, float, float],
        rangefinder_m: float = 0.5,
        armed: bool = True,
        offboard: bool = True,
        landed: bool = False,
        pose_valid: bool = True,
        depth_norm: np.ndarray | None = None,
        healthy: bool = True,
        raycast_stride: int = 4,
    ) -> dict:
        self._lib.ego_planner_on_estimator_health(self._h, float(t_s), 1 if healthy else 0)
        p_arr = (ctypes.c_double * 3)(float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]))
        q_arr = (ctypes.c_double * 4)(
            float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
        )
        if depth_norm is not None:
            d_c = np.ascontiguousarray(depth_norm, dtype=np.float32)
            h, w = d_c.shape[-2], d_c.shape[-1]
            self._lib.ego_planner_on_depth_frame(
                self._h,
                float(t_s),
                d_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(w),
                int(h),
                72.0,
                50.0,
                5.0,
                p_arr,
                q_arr,
                int(raycast_stride),
            )
        else:
            self._lib.ego_planner_on_depth_timestamp(self._h, float(t_s))

        st = _CState()
        st.pos_ned[:] = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
        st.vel_ned[:] = [float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])]
        st.quat_wxyz[:] = [
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        ]
        st.rangefinder_m = float(rangefinder_m)
        st.armed = 1 if armed else 0
        st.offboard = 1 if offboard else 0
        st.landed = 1 if landed else 0
        st.pose_valid = 1 if pose_valid else 0

        out = _COutput()
        self._lib.ego_planner_step(self._h, float(t_s), ctypes.byref(st), ctypes.byref(out))
        return {
            "publish_offboard": bool(out.publish_offboard),
            "publish_arm_cmd": bool(out.publish_arm_cmd),
            "publish_offboard_switch": bool(out.publish_offboard_switch),
            "publish_land_cmd": bool(out.publish_land_cmd),
            "vel_ned": np.array([out.vel_ned[0], out.vel_ned[1], out.vel_ned[2]], dtype=np.float32),
            "yawspeed_rad_s": float(out.yawspeed_rad_s),
            "body_action": np.array(
                [out.body_action[0], out.body_action[1], out.body_action[2], out.body_action[3]],
                dtype=np.float32,
            ),
            "next_checkpoint": int(out.next_checkpoint),
            "phase": out.phase.decode("utf-8"),
            "why": out.why.decode("utf-8"),
            "result": out.result.decode("utf-8"),
            "active_leg": out.active_leg.decode("utf-8"),
            "active_primitive": out.active_primitive.decode("utf-8"),
        }

    def act_sim_single(
        self,
        pos_ned: np.ndarray,
        vel_ned: np.ndarray,
        quat_wxyz: np.ndarray,
        depth_norm: np.ndarray | None = None,
        dt_s: float = 0.10,
        raycast_stride: int = 4,
    ) -> tuple[np.ndarray, dict]:
        if not self._ReadyForSimFlight:
            self.reset(ready_for_sim_flight=True)
        self._sim_t += dt_s
        info = self.step_raw(
            t_s=self._sim_t,
            pos_ned=pos_ned,
            vel_ned=vel_ned,
            quat_wxyz=quat_wxyz,
            rangefinder_m=max(0.04, float(-pos_ned[2])),
            armed=True,
            offboard=True,
            landed=False,
            pose_valid=True,
            depth_norm=depth_norm,
            healthy=True,
            raycast_stride=raycast_stride,
        )
        return info["body_action"], info
