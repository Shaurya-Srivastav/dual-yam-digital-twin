"""Trajectory generation: minimum-jerk segments in joint space and along Cartesian lines,
spline blends through via points, strict IK at every sample, and limit-based time scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from sim_station import layout as L
from sim_station.kin import ArmKin, rot_error


class PlanError(RuntimeError):
    pass


@dataclass
class Traj:
    q: np.ndarray            # (N, 6) joint targets at control rate
    qd: np.ndarray           # (N, 6) joint velocity feed-forward
    dt: float
    duration: float
    length_m: float = 0.0

    def __len__(self) -> int:
        return len(self.q)


def min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def trapezoid(tau: np.ndarray, ramp: float = 0.18) -> np.ndarray:
    """Progress 0..1 with constant speed between short smooth ramps (for guarded descents and sweeps)."""
    tau = np.clip(tau, 0.0, 1.0)
    r = min(ramp, 0.45)
    peak = 1.0 / (1.0 - r)                       # cruise speed that makes the area under the profile equal 1
    u = np.empty_like(tau)
    a = tau < r
    c = (tau >= r) & (tau <= 1.0 - r)
    d = tau > 1.0 - r
    u[a] = 0.5 * peak * tau[a] ** 2 / r
    u[c] = 0.5 * peak * r + peak * (tau[c] - r)
    u[d] = 1.0 - 0.5 * peak * (1.0 - tau[d]) ** 2 / r
    return u


def _sample(path_u: np.ndarray, path_q: np.ndarray, T: float, dt: float, profile: str = "minjerk") -> Traj:
    n = max(int(np.ceil(T / dt)), 1)
    tau = np.minimum(np.arange(n + 1) * dt / T, 1.0)
    u = min_jerk(tau) if profile == "minjerk" else trapezoid(tau)
    q = np.stack([np.interp(u, path_u, path_q[:, j]) for j in range(6)], axis=1)
    qd = np.zeros_like(q)
    qd[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    qd[0] = qd[-1] = 0.0
    return Traj(q=q, qd=qd, dt=dt, duration=n * dt)


def duration_for(dq_per_u: np.ndarray, length_m: float, v_cart: float, speed_scale: float, min_T: float = 0.25,
                 vmax: np.ndarray = L.MAX_JOINT_VEL, amax: np.ndarray = L.MAX_JOINT_ACC) -> float:
    """Peak speed of a min-jerk profile is 1.875 L / T, peak accel 5.7735 L / T^2."""
    mx = np.max(np.abs(dq_per_u) / vmax) if dq_per_u.size else 0.0
    T_v = 1.875 * mx
    ma = np.max(np.abs(dq_per_u) / amax) if dq_per_u.size else 0.0
    T_a = np.sqrt(5.7735 * ma)
    T_c = 1.875 * length_m / max(v_cart, 1e-3) if length_m > 0 else 0.0
    T_ca = np.sqrt(5.7735 * length_m / L.CARRY_ACCEL_M_S2) if length_m > 0 else 0.0
    return max(T_v, T_a, T_c, T_ca, min_T) / max(speed_scale, 1e-3)


def joint_move(q0: np.ndarray, q1: np.ndarray, dt: float = L.CONTROL_DT, speed_scale: float = 1.0,
               min_T: float = 0.3, v_scale: float = 0.8) -> Traj:
    dq = np.asarray(q1, float) - np.asarray(q0, float)
    T = duration_for(dq, 0.0, 1.0, speed_scale, min_T, vmax=L.MAX_JOINT_VEL * v_scale, amax=L.MAX_JOINT_ACC * v_scale)
    return _sample(np.array([0.0, 1.0]), np.vstack([q0, q1]), T, dt)


def via_move(qs: Sequence[np.ndarray], dt: float = L.CONTROL_DT, speed_scale: float = 1.0, min_T: float = 0.4,
             v_scale: float = 0.8) -> Traj:
    """Smooth spline through joint waypoints (no stop at via points), min-jerk timing along the chord length."""
    qs = [np.asarray(q, float) for q in qs]
    if len(qs) == 2:
        return joint_move(qs[0], qs[1], dt, speed_scale, min_T, v_scale)
    chord = np.array([np.linalg.norm(qs[i + 1] - qs[i]) for i in range(len(qs) - 1)])
    keep = np.concatenate([[True], chord > 1e-6])
    qs = [q for q, k in zip(qs, keep) if k]
    chord = chord[chord > 1e-6]
    s = np.concatenate([[0.0], np.cumsum(chord)])
    spline = CubicSpline(s, np.vstack(qs), bc_type="clamped")
    u_grid = np.linspace(0.0, 1.0, max(int(s[-1] / 0.01), 40))
    path_q = spline(u_grid * s[-1])
    dq_du = np.gradient(path_q, u_grid, axis=0)
    T = duration_for(dq_du, 0.0, 1.0, speed_scale, min_T, vmax=L.MAX_JOINT_VEL * v_scale, amax=L.MAX_JOINT_ACC * v_scale)
    return _sample(u_grid, path_q, T, dt)


def _slerp_R(R0: np.ndarray, R1: np.ndarray, u: float) -> np.ndarray:
    Rrel = R0.T @ R1
    c = (np.trace(Rrel) - 1.0) * 0.5
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if ang < 1e-9:
        return R0.copy()
    axis = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / (2.0 * np.sin(ang))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Ru = np.eye(3) + np.sin(ang * u) * K + (1 - np.cos(ang * u)) * K @ K
    return R0 @ Ru


def cart_path(kin: ArmKin, poses: Sequence[np.ndarray], q_seed: np.ndarray, step_m: float = 0.004, step_rad: float = 0.05,
              posture: Optional[np.ndarray] = None, max_joint_step: float = 0.12) -> Tuple[np.ndarray, np.ndarray, float]:
    """Dense joint path following a poly-line of world TCP poses (straight lines, slerped orientation).

    Returns (u grid, joint path, cartesian length)."""
    pts: List[np.ndarray] = []
    for a, b in zip(poses[:-1], poses[1:]):
        L_t = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
        L_r = rot_error(a[:3, :3], b[:3, :3])
        n = max(int(np.ceil(max(L_t / step_m, L_r / step_rad))), 1)
        for i in range(0 if not pts else 1, n + 1):
            u = i / n
            T = np.eye(4)
            T[:3, :3] = _slerp_R(a[:3, :3], b[:3, :3], u)
            T[:3, 3] = (1 - u) * a[:3, 3] + u * b[:3, 3]
            pts.append(T)
    qs = []
    q_prev = np.asarray(q_seed, float)
    for i, T in enumerate(pts):
        r = kin.ik(T, q_prev, posture=posture, max_iters=60)
        if not r.ok:
            r2 = kin.ik(T, q_prev, posture=posture, max_iters=160, dt=0.2)
            if not r2.ok:
                raise PlanError(f"IK failed at sample {i}/{len(pts)} of the line: {r2.reason}")
            r = r2
        if qs and np.max(np.abs(r.q - qs[-1])) > max_joint_step:
            raise PlanError(f"IK branch jump {np.max(np.abs(r.q - qs[-1])):.2f} rad at sample {i}/{len(pts)}")
        qs.append(r.q)
        q_prev = r.q
    path_q = np.array(qs)
    seg = np.array([np.linalg.norm(pts[i + 1][:3, 3] - pts[i][:3, 3]) for i in range(len(pts) - 1)])
    length = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(np.maximum(seg, 1e-9))])
    u_grid = cum / cum[-1]
    return u_grid, path_q, length


def cart_move(kin: ArmKin, poses: Sequence[np.ndarray], q_seed: np.ndarray, v_cart: float = L.CARRY_SPEED_M_S,
              dt: float = L.CONTROL_DT, speed_scale: float = 1.0, min_T: float = 0.25, posture: Optional[np.ndarray] = None,
              step_m: float = 0.004, max_joint_step: float = 0.12, profile: str = "minjerk") -> Traj:
    u_grid, path_q, length = cart_path(kin, poses, q_seed, step_m=step_m, posture=posture, max_joint_step=max_joint_step)
    dq_du = np.gradient(path_q, u_grid, axis=0) if len(u_grid) > 2 else (path_q[-1] - path_q[0])[None, :]
    if profile == "trapezoid":
        # cruise at v_cart: path time = L / v plus the two ramps, subject to the same joint limits
        T = max(duration_for(dq_du, length, v_cart * 1.875 / 1.0, speed_scale, min_T), length / max(v_cart, 1e-3) * 1.0 + 0.18)
        T = max(T, length / max(v_cart, 1e-3) * (1.0 + 0.2) / speed_scale)
    else:
        T = duration_for(dq_du, length, v_cart, speed_scale, min_T)
    traj = _sample(u_grid, path_q, T, dt, profile=profile)
    traj.length_m = length
    return traj


def concat(*trajs: Traj) -> Traj:
    q = np.concatenate([t.q for t in trajs], axis=0)
    qd = np.concatenate([t.qd for t in trajs], axis=0)
    return Traj(q=q, qd=qd, dt=trajs[0].dt, duration=sum(t.duration for t in trajs), length_m=sum(t.length_m for t in trajs))
