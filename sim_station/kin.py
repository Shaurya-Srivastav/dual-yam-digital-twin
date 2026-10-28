"""Kinematics for one YAM arm: strict damped-least-squares IK in SE(3) on the vendor model.

The runner never solves for joint angles by hand. Every skill hands this module a TCP pose in the world
frame (position + approach axis + jaw axis) and gets joint angles back, or a clear failure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mink
import mujoco
import numpy as np

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from sim_station import layout as L

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")


def frame_from(approach: Sequence[float], jaw: Sequence[float], pos: Sequence[float]) -> np.ndarray:
    """TCP frame: z = approach, y = jaw axis (closing axis), x = y cross z."""
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    y = np.asarray(jaw, dtype=float)
    y = y - (y @ z) * z
    n = np.linalg.norm(y)
    if n < 1e-6:
        raise ValueError("jaw axis parallel to approach")
    y = y / n
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, np.asarray(pos, dtype=float)
    return T


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rot_error(R_goal: np.ndarray, R_now: np.ndarray) -> float:
    c = (np.trace(R_goal.T @ R_now) - 1.0) * 0.5
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


@dataclass
class IKResult:
    ok: bool
    q: np.ndarray
    pos_err: float
    ori_err: float
    iters: int
    reason: str = ""


class ArmKin:
    LIMIT_MARGIN = 0.008

    def __init__(self, arm: str, mount: Optional[np.ndarray] = None, yaw: Optional[float] = None) -> None:
        self.arm = arm
        path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.model.jnt_range[3, 1] = L.JOINT_HIGH[3]
        self.config = mink.Configuration(self.model)
        self.site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
        base, base_yaw = L.MOUNTS[arm]
        self.mount = np.array(base if mount is None else mount, dtype=float)
        self.yaw = float(base_yaw if yaw is None else yaw)
        self.R_wb = rotz(self.yaw)
        self.lo = L.JOINT_LOW + self.LIMIT_MARGIN
        self.hi = L.JOINT_HIGH - self.LIMIT_MARGIN
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.ref_q = np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0])

    # ---- frames ---------------------------------------------------------------
    def world_to_base(self, T_world: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R_wb.T @ T_world[:3, :3]
        T[:3, 3] = self.R_wb.T @ (T_world[:3, 3] - self.mount)
        return T

    def base_to_world(self, T_base: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R_wb @ T_base[:3, :3]
        T[:3, 3] = self.R_wb @ T_base[:3, 3] + self.mount
        return T

    # ---- FK -------------------------------------------------------------------
    def fk_base(self, q6: np.ndarray) -> np.ndarray:
        d = self.data
        d.qpos[:6] = q6
        d.qpos[6:8] = 0.0
        mujoco.mj_kinematics(self.model, d)
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[self.site].reshape(3, 3)
        T[:3, 3] = d.site_xpos[self.site]
        return T

    def fk(self, q6: np.ndarray) -> np.ndarray:
        return self.base_to_world(self.fk_base(q6))

    def jacobian(self, q6: np.ndarray) -> np.ndarray:
        d = self.data
        d.qpos[:6] = q6
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jp, jr, self.site)
        return np.vstack([jp[:, :6], jr[:, :6]])

    # ---- IK -------------------------------------------------------------------
    def ik(self, T_world: np.ndarray, seed: Sequence[float], posture: Optional[Sequence[float]] = None,
           pos_tol: float = 4e-4, ori_tol: float = 6e-3, max_iters: int = 160, dt: float = 0.35,
           posture_cost: float = 2e-3) -> IKResult:
        target = self.world_to_base(T_world)
        q0 = np.zeros(self.model.nq)
        q0[:6] = np.clip(np.asarray(seed, dtype=float), L.JOINT_LOW, L.JOINT_HIGH)
        self.config.update(q0)
        task = mink.FrameTask("grasp_site", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-3)
        task.set_target(mink.SE3.from_matrix(target))
        tasks: List = [task]
        if posture is not None:
            post = mink.PostureTask(self.model, cost=posture_cost)
            qp = np.zeros(self.model.nq)
            qp[:6] = posture
            post.set_target(qp)
            tasks.append(post)
        err = task.compute_error(self.config)
        for it in range(max_iters):
            pe, oe = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
            if pe < pos_tol and oe < ori_tol:
                break
            vel = mink.solve_ik(self.config, tasks, dt, "quadprog", damping=1e-4, limits=self.limits)
            vel[6:] = 0.0
            self.config.integrate_inplace(vel, dt)
            err = task.compute_error(self.config)
        q = np.array(self.config.q[:6])
        pe, oe = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        ok = pe < pos_tol and oe < ori_tol
        reason = ""
        if not ok:
            reason = f"no convergence (pos {pe*1000:.2f} mm, ori {np.degrees(oe):.2f} deg)"
        elif np.any(q < self.lo) or np.any(q > self.hi):
            ok = False
            j = int(np.argmax(np.maximum(self.lo - q, q - self.hi)))
            reason = f"joint {j + 1} at limit ({q[j]:.2f})"
        return IKResult(ok, q, pe, oe, it, reason)

    def ik_multi(self, T_world: np.ndarray, seeds: Sequence[Sequence[float]], posture: Optional[Sequence[float]] = None,
                 **kw) -> IKResult:
        best = None
        for s in seeds:
            r = self.ik(T_world, s, posture, **kw)
            if r.ok:
                return r
            if best is None or r.pos_err + r.ori_err < best.pos_err + best.ori_err:
                best = r
        return best

    def manipulability(self, q6: np.ndarray) -> float:
        J = self.jacobian(q6)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
