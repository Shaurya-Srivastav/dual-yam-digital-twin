"""Physics plant: MuJoCo model + the real MIT-mode controller, the gripper force servo, noisy
joint feedback, the nested-stack bond, the passive spigot, and per-level resets.

The controller side of the system sees the plant only through `read_arm` (encoder position,
velocity, motor torque feedback, gripper width/effort) and `command_arm` / `command_gripper`.
Object poses are exposed for scoring and the debug overlay, never for the runner.
"""

from __future__ import annotations

import copy
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from sim_station import layout as L
from sim_station import scene

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

ARMS = ("cup", "tap")
PARK_Z = -0.6


@dataclass
class Randomization:
    """Monte Carlo draw. All fields are scale factors or offsets applied by `Plant.reset`."""
    friction_scale: float = 1.0
    cup_mass_scale: float = 1.0
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    gravity_model_error: np.ndarray = field(default_factory=lambda: np.zeros(6))
    base_offset: Dict[str, np.ndarray] = field(default_factory=lambda: {"cup": np.zeros(3), "tap": np.zeros(3)})
    base_yaw: Dict[str, float] = field(default_factory=lambda: {"cup": 0.0, "tap": 0.0})
    object_jitter_xy: float = 0.0
    sensor_noise: float = 1.0
    cmd_delay_steps: int = 2
    stack_count: Optional[int] = None
    lever_yaw: Optional[float] = None
    cup_xy: Optional[np.ndarray] = None
    stack_break_scale: float = 1.0

    @staticmethod
    def draw(rng: np.random.Generator, level: int) -> "Randomization":
        r = Randomization()
        r.friction_scale = float(rng.uniform(0.75, 1.25))
        r.cup_mass_scale = float(rng.uniform(0.75, 1.3))
        r.kp_scale = float(rng.uniform(0.9, 1.1))
        r.kd_scale = float(rng.uniform(0.9, 1.1))
        r.gravity_model_error = rng.normal(0.0, 0.02, 6)
        for arm in ARMS:
            r.base_offset[arm] = np.append(rng.normal(0.0, 0.002, 2), 0.0)
            r.base_yaw[arm] = float(rng.normal(0.0, np.radians(0.5)))
        r.object_jitter_xy = 0.008
        r.sensor_noise = float(rng.uniform(0.8, 1.6))
        r.cmd_delay_steps = int(rng.integers(2, 6))
        r.stack_break_scale = float(rng.uniform(0.8, 1.2))
        if level == 3:
            r.stack_count = int(rng.integers(3, L.NEST_MAX + 1))
        if level == 2:
            r.stack_count = L.NEST_DEFAULT
        if level == 4:
            r.lever_yaw = float(rng.uniform(-np.pi, np.pi))
        if level == 5:
            for _ in range(200):
                xy = np.array([rng.uniform(*L.L5_REGION_X), rng.uniform(*L.L5_REGION_Y)])
                if all(np.linalg.norm(xy - c) > rad for c, rad in L.L5_KEEPOUT):
                    r.cup_xy = xy
                    break
        return r


@dataclass
class ArmObs:
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    grip_width: float
    grip_vel: float
    grip_force: float
    t: float


@dataclass
class ArmIdx:
    prefix: str
    qpos: np.ndarray
    dof: np.ndarray
    act: np.ndarray
    g_qpos: np.ndarray
    g_dof: np.ndarray
    g_act: np.ndarray
    site: int
    pad_geoms: List[int]
    body_ids: List[int]


class Plant:
    def __init__(self, seed: int = 0, nominal_model: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        xml = scene.build_xml()
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.nom_model = mujoco.MjModel.from_xml_string(xml)       # the controller's model (never randomized)
        self.nom_data = mujoco.MjData(self.nom_model)
        self.scratch = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        m = self.model

        self.arm: Dict[str, ArmIdx] = {}
        for arm in ARMS:
            p = f"{arm}_"
            jq = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + n)])
            jd = lambda n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + n)])
            ja = lambda n: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, p + n))
            pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p}pad_{s}_{k}") for s in ("left", "right") for k in ("tip", "funnel")]
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p + "mount")
            bodies = [b for b in range(m.nbody) if self._is_descendant(b, root)]
            self.arm[arm] = ArmIdx(
                prefix=p,
                qpos=np.array([jq(f"joint{i}") for i in range(1, 7)]),
                dof=np.array([jd(f"joint{i}") for i in range(1, 7)]),
                act=np.array([ja(f"a{i}") for i in range(1, 7)]),
                g_qpos=np.array([jq("joint7"), jq("joint8")]),
                g_dof=np.array([jd("joint7"), jd("joint8")]),
                g_act=np.array([ja("g7"), ja("g8")]),
                site=int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, p + "grasp_site")),
                pad_geoms=pads,
                body_ids=bodies,
            )
        name2 = lambda t, n: int(mujoco.mj_name2id(m, t, n))
        B, J, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
        self.body = {n: name2(B, n) for n in ("paper_cup", "plastic_cup", "paper_rest", "plastic_rest", "coaster", "dispenser", "lever", "lever_turret")}
        self.free_qadr = {n: int(m.jnt_qposadr[name2(J, f"{n}_free")]) for n in ("paper_cup", "plastic_cup", "paper_rest", "plastic_rest")}
        self.free_dadr = {n: int(m.jnt_dofadr[name2(J, f"{n}_free")]) for n in ("paper_cup", "plastic_cup", "paper_rest", "plastic_rest")}
        self.lever_q = int(m.jnt_qposadr[name2(J, "lever")])
        self.lever_dof = int(m.jnt_dofadr[name2(J, "lever")])
        self.yaw_q = int(m.jnt_qposadr[name2(J, "lever_yaw")])
        self.yaw_dof = int(m.jnt_dofadr[name2(J, "lever_yaw")])
        self.disp_q = [int(m.jnt_qposadr[name2(J, "disp_x")]), int(m.jnt_qposadr[name2(J, "disp_y")])]
        self.stream_geom = name2(G, "stream")
        self.geoms_of: Dict[str, List[int]] = {}
        for name in ("paper_cup", "plastic_cup", "paper_rest", "plastic_rest", "coaster", "dispenser"):
            self.geoms_of[name] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body[name]]
        self._arm_geoms: Dict[str, set] = {
            arm: {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith((f"{arm}_col", f"{arm}_pad"))}
            for arm in ("cup", "tap")}
        self.geom_type0 = m.geom_contype.copy()
        self.geom_aff0 = m.geom_conaffinity.copy()
        self.geom_rgba0 = m.geom_rgba.copy()
        self.band_geom: Dict[str, List[int]] = {}
        self.full_geom: Dict[str, List[int]] = {}
        self.vis_geoms: Dict[str, List[List[int]]] = {}
        for kind in ("paper", "plastic"):
            self.band_geom[kind] = [name2(G, f"{kind}_rest_band{i}") for i in range(scene.STACK_COUNT_MAX)]
            self.full_geom[kind] = [name2(G, f"{kind}_rest_full{i}") for i in range(scene.STACK_COUNT_MAX)]
            vis = []
            for i in range(scene.STACK_COUNT_MAX):
                ids = [name2(G, f"{kind}_rest_vis{i}"), name2(G, f"{kind}_rest_lip{i}")]
                inn = name2(G, f"{kind}_rest_in{i}")
                if inn >= 0:
                    ids.append(inn)
                vis.append(ids)
            self.vis_geoms[kind] = vis
        self.rest_mass0 = {k: float(m.body_mass[self.body[f"{k}_rest"]]) for k in ("paper", "plastic")}

        # controller state
        self.ctl = {a: self._new_ctl() for a in ARMS}
        self.cmd_queue: Dict[str, deque] = {a: deque() for a in ARMS}
        self.rand = Randomization()
        self.kp = {a: L.KP_ARM.copy() for a in ARMS}
        self.grav_adapt = {a: np.ones(6) for a in ARMS}
        self.stack_is_obstacle = True       # the stack stands on the coaster, off the spout corridor; links and housing must go around it
        self.kd = {a: L.KD_ARM.copy() for a in ARMS}
        self.friction_comp = False
        self._grav = {a: np.zeros(6) for a in ARMS}
        self._grav_tick = 0
        self.t_obs = 0.0
        self._noise_rng = np.random.default_rng(seed + 17)
        self.sensor_noise = 1.0
        self.last_tau = {a: np.zeros(6) for a in ARMS}

        # world / task state
        self.level = 1
        self.material = "paper"
        self.layout = L.level_layout(1)
        self.active_cup = "paper_cup"
        self.active_rest = "paper_rest"
        self.stack_count = L.NEST_DEFAULT
        self.tether = {"on": False}
        self.pour_s = 7.0
        self.flow_time = 0.0
        self.flow_time_with_cup = 0.0
        self.flow_events: List[Tuple[str, float]] = []
        self.violations: dict = {}
        self._audit_tick = 0
        self._flow_on = False
        self.peak_pad_force = 0.0
        self.contact_log: List[str] = []
        self.history = deque(maxlen=600)
        self._hist_tick = 0
        self.steps = 0
        self.dispenser_rest = np.zeros(2)
        self.reset(1, "paper")

    # ------------------------------------------------------------------ utils
    def _is_descendant(self, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def _new_ctl(self) -> dict:
        return dict(
            q_cmd=np.zeros(6), qd_cmd=np.zeros(6), tau_ff=np.zeros(6),
            grip_target=L.GRIPPER_MAX_OPEN_M, grip_ref=L.GRIPPER_MAX_OPEN_M, grip_cap=L.FORCE_CAP_PAPER_CUP_N,
            grip_rate=L.GRIPPER_RATE_M_S,
        )

    @property
    def time(self) -> float:
        return float(self.data.time)

    def arm_q(self, arm: str) -> np.ndarray:
        return np.array(self.data.qpos[self.arm[arm].qpos])

    def arm_qd(self, arm: str) -> np.ndarray:
        return np.array(self.data.qvel[self.arm[arm].dof])

    def grip_width_true(self, arm: str) -> float:
        a = self.arm[arm]
        return float(self.data.qpos[a.g_qpos[0]] + self.data.qpos[a.g_qpos[1]])

    def tcp_true(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        a = self.arm[arm]
        return np.array(self.data.site_xpos[a.site]), np.array(self.data.site_xmat[a.site]).reshape(3, 3)

    def body_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        b = self.body[name] if name in self.body else int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
        return np.array(self.data.xpos[b]), np.array(self.data.xmat[b]).reshape(3, 3)

    def lever_angle(self) -> float:
        return float(self.data.qpos[self.lever_q])

    def lever_yaw(self) -> float:
        return float(self.data.qpos[self.yaw_q])

    # --------------------------------------------------------------- commands
    def command_arm(self, arm: str, q: np.ndarray, qd: Optional[np.ndarray] = None, tau_ff: Optional[np.ndarray] = None) -> None:
        item = (np.array(q, dtype=float), np.zeros(6) if qd is None else np.array(qd, dtype=float),
                np.zeros(6) if tau_ff is None else np.array(tau_ff, dtype=float))
        queue = self.cmd_queue[arm]
        queue.append(item)

    def command_gripper(self, arm: str, width_m: float, cap_n: Optional[float] = None, rate: Optional[float] = None) -> None:
        c = self.ctl[arm]
        c["grip_target"] = float(np.clip(width_m, 0.0, L.GRIPPER_MAX_OPEN_M))
        if cap_n is not None:
            self.set_grip_cap(arm, cap_n)
        if rate is not None:
            c["grip_rate"] = float(rate)

    def set_grip_cap(self, arm: str, cap_n: float) -> None:
        c = self.ctl[arm]
        c["grip_cap"] = float(cap_n)
        for act in self.arm[arm].g_act:
            self.model.actuator_forcerange[act] = [-cap_n, cap_n]

    # ------------------------------------------------------------------ stepping
    def _gravity_nominal(self) -> None:
        nd, nm = self.nom_data, self.nom_model
        for arm in ARMS:
            a = self.arm[arm]
            nd.qpos[a.qpos] = self.data.qpos[a.qpos]
            nd.qpos[a.g_qpos] = 0.0
        nd.qvel[:] = 0.0
        nd.qacc[:] = 0.0
        mujoco.mj_kinematics(nm, nd)
        mujoco.mj_comPos(nm, nd)
        out = np.zeros(nm.nv)
        mujoco.mj_rne(nm, nd, 0, out)
        for arm in ARMS:
            self._grav[arm] = out[self.arm[arm].dof].copy()

    def _apply_controls(self) -> None:
        d, m = self.data, self.model
        if self._grav_tick % 2 == 0:
            self._gravity_nominal()
        self._grav_tick += 1
        for arm in ARMS:
            a, c = self.arm[arm], self.ctl[arm]
            queue = self.cmd_queue[arm]
            if len(queue) > self.rand.cmd_delay_steps:
                q_cmd, qd_cmd, tau = queue.popleft()
                c["q_cmd"], c["qd_cmd"], c["tau_ff"] = q_cmd, qd_cmd, tau
            qd = d.qvel[a.dof]
            grav = self._grav[arm] * L.GRAV_COMP_FACTOR * self.grav_adapt[arm] * (1.0 + self.rand.gravity_model_error)
            fric = L.COULOMB_FRICTION * np.tanh(qd / 0.04) if self.friction_comp else 0.0
            tau = c["tau_ff"] + grav + fric
            kp, kd = self.kp[arm], self.kd[arm]
            d.ctrl[a.act] = c["q_cmd"] + (kd / kp) * c["qd_cmd"] + tau / kp
            # gripper: rate-limited reference width, force-saturated position servo
            err = c["grip_target"] - c["grip_ref"]
            step = c["grip_rate"] * self.dt
            c["grip_ref"] += max(-step, min(step, err))
            half = float(np.clip(0.5 * c["grip_ref"], 0.0, L.FINGER_MAX_M))
            d.ctrl[a.g_act[0]] = half
            d.ctrl[a.g_act[1]] = half

    def _rest_pinned_by_tap(self) -> bool:
        """True when both tap fingertip pads are pinching the remaining stack with enough normal force
        that Coulomb static friction holds the stack against the tether pull (preventing soft-contact viscous drift)."""
        m, d = self.model, self.data
        a = self.arm["tap"]
        gf = 0.5 * (abs(float(d.actuator_force[a.g_act[0]])) + abs(float(d.actuator_force[a.g_act[1]])))
        if gf < 8.0:
            return False
        if not hasattr(self, "_tap_pad_lr"):
            left = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("tap_pad_left_tip", "tap_pad_left_funnel")}
            right = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("tap_pad_right_tip", "tap_pad_right_funnel")}
            self._tap_pad_lr = (left, right)
        left_g, right_g = self._tap_pad_lr
        rest_g = set(self.geoms_of[self.active_rest])
        hit_l = hit_r = False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 2e-4:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in left_g and g2 in rest_g) or (g2 in left_g and g1 in rest_g):
                hit_l = True
            elif (g1 in right_g and g2 in rest_g) or (g2 in right_g and g1 in rest_g):
                hit_r = True
            if hit_l and hit_r:
                return True
        return False

    def _tether_forces(self) -> None:
        d = self.data
        tb = self.tether
        d.xfrc_applied[self.body[self.active_cup]] = 0.0
        d.xfrc_applied[self.body[self.active_rest]] = 0.0
        if tb.get("releasing"):
            # The bond is gone, but the cup is still inside the stack mesh. Turning collision back on
            # there throws it out of the fingers. Wait until it has lifted clear, then collide normally.
            top, rest = self.body[self.active_cup], self.body[self.active_rest]
            p1 = d.xpos[top]
            p2 = d.xpos[rest]
            R2 = d.xmat[rest].reshape(3, 3)
            nested = p2 + R2 @ tb["rel_p"]
            vert = float(p1[2] - nested[2])
            horiz = float(np.linalg.norm(p1[:2] - nested[:2]))
            # Stay non-colliding until the cup is actually out of the nest. Turning the mesh on
            # while it still overlaps the cups below launches it out of the fingers.
            # Vertical clearance during the in-place rise still overlaps a short stack and the
            # contact impulse blows the stack joint up. Wait until the cup has slid off to the side.
            far = float(np.linalg.norm(p1 - nested))
            # A tall vertical rise still overlaps a short nest. Turning the mesh on there
            # blows the stack joint. Sideways clearance, or the cup actually falling, is the gate.
            if horiz > 0.07 or vert < -0.012 or far > 0.15:
                tb["releasing"] = False
                self._cup_collides_normally(self.active_cup)
            return
        if not tb["on"]:
            return
        top, rest = self.body[self.active_cup], self.body[self.active_rest]
        p1, R1 = d.xpos[top], d.xmat[top].reshape(3, 3)
        p2, R2 = d.xpos[rest], d.xmat[rest].reshape(3, 3)
        want_p = p2 + R2 @ tb["rel_p"]
        dp = p1 - want_p
        v1 = d.qvel[self.free_dadr[self.active_cup]: self.free_dadr[self.active_cup] + 3]
        v2 = d.qvel[self.free_dadr[self.active_rest]: self.free_dadr[self.active_rest] + 3]
        f = -np.array([tb["k_lat"] * dp[0], tb["k_lat"] * dp[1], tb["k"] * dp[2]]) - tb["c"] * (v1 - v2)
        R_des = R2 @ tb["rel_R"]
        e = 0.5 * (np.cross(R1[:, 0], R_des[:, 0]) + np.cross(R1[:, 1], R_des[:, 1]) + np.cross(R1[:, 2], R_des[:, 2]))
        w1 = R1 @ d.qvel[self.free_dadr[self.active_cup] + 3: self.free_dadr[self.active_cup] + 6]
        w2 = R2 @ d.qvel[self.free_dadr[self.active_rest] + 3: self.free_dadr[self.active_rest] + 6]
        tq = tb["kr"] * e - tb["cr"] * (w1 - w2)
        d.xfrc_applied[top, :3] = f
        d.xfrc_applied[top, 3:] = tq
        if self._rest_pinned_by_tap() and float(np.linalg.norm(f)) < 22.0:
            d.xfrc_applied[rest, 0] = 0.0
            d.xfrc_applied[rest, 1] = 0.0
            d.xfrc_applied[rest, 2] = min(0.0, float(-f[2]))
            d.xfrc_applied[rest, 3:] = 0.0
            da_r = self.free_dadr[self.active_rest]
            d.qvel[da_r: da_r + 3] = 0.0
        else:
            d.xfrc_applied[rest, :3] = -f
            d.xfrc_applied[rest, 3:] = -tq
        # Fatigue: only reversing axial (vertical) motion loosens nested cups. Count the amplitude of each swing larger than
        # a hysteresis band; slow drift and contact-solver jitter do not count.
        z = float(dp[2])
        band = 2.5e-4
        if tb["dir"] == 0:
            tb["ext"] = z
            tb["dir"] = 1 if z > tb["z0"] + band else (-1 if z < tb["z0"] - band else 0)
        elif tb["dir"] > 0:
            if z > tb["ext"]:
                tb["ext"] = z
            elif tb["ext"] - z > band:
                tb["path"] += tb["ext"] - tb["last_turn"]
                tb["last_turn"] = tb["ext"]
                tb["ext"] = z
                tb["dir"] = -1
        else:
            if z < tb["ext"]:
                tb["ext"] = z
            elif z - tb["ext"] > band:
                tb["path"] += tb["last_turn"] - tb["ext"]
                tb["last_turn"] = tb["ext"]
                tb["ext"] = z
                tb["dir"] = 1
        tb["force"] = float(np.linalg.norm(f))
        limit = tb["break_n"] * float(np.exp(-tb["path"] / L.STACK_FATIGUE_LEN_M))
        if tb["path"] > 0.002:
            limit = max(6.8, limit)
        if tb["force"] > limit and not tb.get("releasing"):
            scale = limit / max(tb["force"], 1e-9)
            d.xfrc_applied[top, :3] *= scale
            d.xfrc_applied[top, 3:] *= scale
            d.xfrc_applied[rest, :3] *= scale
            d.xfrc_applied[rest, 3:] *= scale
            tb["force"] = float(limit)
            tb["on"] = False
            tb["releasing"] = True
            tb["broke_at"] = self.time
            tb["broke_force"] = float(limit)

    STACK_UPRIGHT_K = 4.0          # N m/rad: a nested stack stands on a base much wider than a single cup's grip would suggest
    STACK_UPRIGHT_C = 0.08         # N m s/rad

    def _stabilize_stacks(self) -> None:
        """A stack of cups must never fall over: a restoring torque about its centre keeps it upright (it can still be pushed
        across the mat, and it still tilts a few degrees under a push)."""
        m, d = self.model, self.data
        for name in ("paper_rest", "plastic_rest"):
            b = self.body[name]
            if m.body_gravcomp[b] > 0.0:
                continue                                            # parked below the table
            R = d.xmat[b].reshape(3, 3)
            zb = R[:, 2]
            da = self.free_dadr[name]
            w = R @ d.qvel[da + 3: da + 6]
            # Scale stiffness and damping with the stack's own moment of inertia so a 2-cup
            # remainder (ixx ~ 3.6e-5) is never over-damped past the 1 kHz Euler stability limit.
            ixx = float(max(m.body_inertia[b][0], 2.5e-5))
            k = self.STACK_UPRIGHT_K
            c = min(self.STACK_UPRIGHT_C, 1.5 * float(np.sqrt(k * ixx)))
            tq = k * np.cross(zb, np.array([0.0, 0.0, 1.0])) - c * np.array([w[0], w[1], 0.0])
            d.xfrc_applied[b, 3:] += tq

    def step(self, n: int = 1) -> None:
        m, d = self.model, self.data
        for _ in range(n):
            self._apply_controls()
            self._tether_forces()
            self._stabilize_stacks()
            mujoco.mj_step(m, d)
            self.steps += 1
        self._post_batch(n)

    def _post_batch(self, n: int = 1) -> None:
        d = self.data
        lever = float(d.qpos[self.lever_q])
        flowing = lever > L.LEVER_FLOW_THRESHOLD
        if flowing != self._flow_on:
            self._flow_on = flowing
            self.flow_events.append(("open" if flowing else "close", self.time))
        if flowing:
            self.flow_time += n * self.dt
            if self._cup_under_outlet():
                self.flow_time_with_cup += n * self.dt
        self.model.geom_rgba[self.stream_geom, 3] = 0.55 if flowing else 0.0
        self.last_tau = {a: np.array(d.actuator_force[self.arm[a].act]) for a in ARMS}
        self._audit_tick += 1
        if self._audit_tick % 3 == 0:
            self._audit_contacts()
        self._hist_tick += 1
        if self._hist_tick % 16 == 0:
            self.history.append(self._history_sample())

    def _audit_contacts(self) -> None:
        """Referee: record contacts that must never happen (arm on arm, arm on the dispenser body, arm on the table)."""
        m, d = self.model, self.data
        if not hasattr(self, "_audit_sets"):
            names = lambda ns: {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ns}
            self._audit_sets = dict(
                dispenser=names(("tank_hit", "valve_barrel", "valve_body", "spout_tip")),
                table=names(("tabletop",)), mat=names(("cutting_mat",)))
        S = self._audit_sets
        cup_g = set(self.geoms_of[self.active_cup]) | set(self.geoms_of[self.active_rest])
        spout_g = S["dispenser"] | {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("lever_stem", "lever_paddle")}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0.0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in cup_g and g2 in spout_g) or (g2 in cup_g and g1 in spout_g):         # a cup must never touch the spout, valve or handle
                key = ("cup_dispenser", min(g1, g2), max(g1, g2))
                if key not in self.violations:
                    self.violations[key] = dict(t=self.time, kind="cup_dispenser", a=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                                b=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2))
            for a, b in ((g1, g2), (g2, g1)):
                for arm in ARMS:
                    if a in self._arm_geoms[arm] and b in cup_g and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a) or "").startswith(f"{arm}_pad"):
                        key = ("arm_stack", a, b)            # links and housing must never touch a cup or the stack; only the fingertips may
                        if key not in self.violations:
                            self.violations[key] = dict(t=self.time, kind="arm_stack", a=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a), b=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b))
                    if a in self._arm_geoms[arm]:
                        other = "tap" if arm == "cup" else "cup"
                        kind = None
                        if b in self._arm_geoms[other]:
                            kind = "arm_arm"
                        elif b in S["dispenser"]:
                            kind = "arm_dispenser"
                        elif b in S["table"]:
                            kind = "arm_table"
                        elif b in S["mat"]:
                            kind = "arm_mat"
                        if kind is not None:
                            key = (kind, min(a, b), max(a, b))
                            if key not in self.violations:
                                self.violations[key] = dict(t=self.time, kind=kind, a=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a),
                                                            b=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b))

    def fatal_violations(self) -> list:
        return [v for v in self.violations.values() if v["kind"] in ("arm_arm", "arm_dispenser", "arm_table", "cup_dispenser", "arm_stack")]

    def run_time(self, seconds: float, control_cb=None) -> None:
        steps = int(round(seconds / self.dt))
        for _ in range(steps):
            if control_cb is not None:
                control_cb()
            self.step(1)

    # ------------------------------------------------------------------ sensors
    def read_arm(self, arm: str) -> ArmObs:
        a, d = self.arm[arm], self.data
        s = self.sensor_noise
        rng = self._noise_rng
        q = np.array(d.qpos[a.qpos]) + rng.normal(0.0, 1.2e-4 * s, 6)
        qd = np.array(d.qvel[a.dof]) + rng.normal(0.0, 1.5e-3 * s, 6)
        tau = np.array(d.actuator_force[a.act]) + rng.normal(0.0, 1.2e-2 * s, 6)
        width = float(d.qpos[a.g_qpos[0]] + d.qpos[a.g_qpos[1]]) + rng.normal(0.0, 8e-5 * s)
        gv = float(d.qvel[a.g_dof[0]] + d.qvel[a.g_dof[1]])
        gf = float(0.5 * (abs(d.actuator_force[a.g_act[0]]) + abs(d.actuator_force[a.g_act[1]])))
        return ArmObs(q, qd, tau, width, gv, gf, self.time)

    def gravity_model(self, arm: str, q: np.ndarray, adapted: bool = True) -> np.ndarray:
        """Static gravity torque from the controller's nominal model at an arbitrary q (times the learned correction)."""
        nd, nm = self.nom_data, self.nom_model
        a = self.arm[arm]
        nd.qpos[a.qpos] = q
        nd.qvel[:] = 0.0
        nd.qacc[:] = 0.0
        mujoco.mj_kinematics(nm, nd)
        mujoco.mj_comPos(nm, nd)
        out = np.zeros(nm.nv)
        mujoco.mj_rne(nm, nd, 0, out)
        g = out[a.dof].copy()
        return g * self.grav_adapt[arm] if adapted else g

    def jacobian_site(self, arm: str) -> np.ndarray:
        a = self.arm[arm]
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, a.site)
        return np.vstack([jp[:, a.dof], jr[:, a.dof]])

    # -------------------------------------------------------------- truth queries
    def outlet_xy_true(self) -> np.ndarray:
        dxy = self.data.xpos[self.body["dispenser"]][:2]
        return np.array([dxy[0] + L.FAUCET_POS[0] + L.VALVE_LOCAL_X, dxy[1] + L.FAUCET_POS[1]])

    def _cup_under_outlet(self) -> bool:
        p, R = self.body_pose(self.active_cup)
        return bool(np.linalg.norm(p[:2] - self.outlet_xy_true()) < 0.016 and R[2, 2] > 0.9
                    and p[2] + 0.5 * L.CUP_HEIGHT < L.MOUTH_OUTLET_Z + 0.002)

    def cup_state(self) -> dict:
        p, R = self.body_pose(self.active_cup)
        up = float(R[2, 2])
        return dict(xyz=p, up=up, axis=R[:, 2].copy(), upright=up > 0.94)

    def pad_force(self, arm: str) -> float:
        """Total pad-normal force on objects (truth, for telemetry/evaluation only)."""
        a = self.arm[arm]
        pads = set(a.pad_geoms)
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in pads or c.geom2 in pads:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])
        return float(total)

    def touching(self, arm: str, body_name: str) -> bool:
        a = self.arm[arm]
        pads = set(a.pad_geoms)
        bid = self.body[body_name] if body_name in self.body else -1
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in pads and self.model.geom_bodyid[g2] == bid) or (g2 in pads and self.model.geom_bodyid[g1] == bid):
                return True
        return False

    def _history_sample(self) -> dict:
        row = {"t": self.time, "lever": self.lever_angle(), "yaw": self.lever_yaw()}
        for arm in ARMS:
            a = self.arm[arm]
            tcp, _ = self.tcp_true(arm)
            v = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, a.site, v, 0)
            row[f"{arm}_speed"] = float(np.linalg.norm(v[3:]))
            row[f"{arm}_grip_f"] = float(0.5 * (abs(self.data.actuator_force[a.g_act[0]]) + abs(self.data.actuator_force[a.g_act[1]])))
            row[f"{arm}_grip_cap"] = float(self.ctl[arm]["grip_cap"])
            row[f"{arm}_width_mm"] = float(1000.0 * self.grip_width_true(arm))
        return row

    # ------------------------------------------------------- snapshot / restore
    def save_state(self) -> dict:
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        s = np.zeros(size)
        mujoco.mj_getState(self.model, self.data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
        return dict(
            state=s, ctl=copy.deepcopy(self.ctl), tether=copy.deepcopy(self.tether), flow_time=self.flow_time,
            flow_time_with_cup=self.flow_time_with_cup, flow_events=list(self.flow_events),
            ctrl=self.data.ctrl.copy(), forcerange=self.model.actuator_forcerange.copy(),
        )

    def restore_state(self, snap: dict) -> None:
        mujoco.mj_setState(self.model, self.data, snap["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        self.ctl = copy.deepcopy(snap["ctl"])
        self.tether = copy.deepcopy(snap["tether"])
        self.flow_time = snap["flow_time"]
        self.flow_time_with_cup = snap["flow_time_with_cup"]
        self.flow_events = list(snap["flow_events"])
        self.data.ctrl[:] = snap["ctrl"]
        self.model.actuator_forcerange[:] = snap["forcerange"]
        for arm in ARMS:
            self.cmd_queue[arm].clear()
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------- collision queries
    def collisions_for(self, q: Dict[str, np.ndarray], grip: Optional[Dict[str, float]] = None) -> List[Tuple[str, str]]:
        """Contacts of the arms with static/passive world and each other for given joint vectors (objects ignored)."""
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0.0
        for arm in ARMS:
            a = self.arm[arm]
            if arm in q:
                s.qpos[a.qpos] = q[arm]
            w = (grip or {}).get(arm, 0.0776)
            s.qpos[a.g_qpos] = 0.5 * w
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_collision(self.model, s)
        out = []
        ignore = set(self.geoms_of["paper_cup"] + self.geoms_of["plastic_cup"] + self.geoms_of["paper_rest"] +
                     self.geoms_of["plastic_rest"] + self.geoms_of["coaster"])
        # what is left of a stack is an obstacle for everything except the fingertips that are meant to touch it
        stack_g = set()
        if self.stack_is_obstacle and self.level in (2, 3) and self.model.body_gravcomp[self.body[self.active_rest]] == 0.0:
            stack_g = {g for g in self.geoms_of[self.active_rest] if self.model.geom_contype[g] or self.model.geom_conaffinity[g]}
            ignore -= stack_g
        lever_geoms = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("lever_stem", "lever_paddle")}
        for i in range(s.ncon):
            c = s.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in ignore or g2 in ignore or g1 in lever_geoms or g2 in lever_geoms:
                continue
            if stack_g and ((g1 in stack_g and (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or "").find("_pad") >= 0) or
                            (g2 in stack_g and (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or "").find("_pad") >= 0)):
                continue
            out.append((mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1),
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)))
        return out

    def cup_hits_arm(self, arm: str, q: np.ndarray, grip: float, cup_xy: np.ndarray) -> bool:
        """Would an upright cup standing at cup_xy touch this arm at joint vector q? This is the planner's model query
        (the arm's and the cup's CAD), evaluated on a scratch copy: the true cup pose is overwritten, never read."""
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0.0
        a = self.arm[arm]
        s.qpos[a.qpos] = q
        s.qpos[a.g_qpos] = 0.5 * grip
        qa = self.free_qadr[self.active_cup]
        s.qpos[qa: qa + 7] = [float(cup_xy[0]), float(cup_xy[1]), L.MAT_SURFACE_Z + 0.5 * L.CUP_HEIGHT + 0.0006, 1, 0, 0, 0]
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_collision(self.model, s)
        cup_g = set(self.geoms_of[self.active_cup])
        arm_g = self._arm_geoms[arm]
        for i in range(s.ncon):
            c = s.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in cup_g and g2 in arm_g) or (g2 in cup_g and g1 in arm_g):
                return True
        return False

    # ------------------------------------------------------------------- reset
    def _park(self, name: str) -> None:
        """Hide a free body below the table and remove its collisions."""
        qa, da = self.free_qadr[name], self.free_dadr[name]
        slot = {"paper_cup": (0.0, -1.0), "plastic_cup": (0.3, -1.0), "paper_rest": (0.0, -1.3), "plastic_rest": (0.3, -1.3)}[name]
        self.data.qpos[qa: qa + 7] = [slot[0], slot[1], PARK_Z, 1, 0, 0, 0]
        self.data.qvel[da: da + 6] = 0.0
        self.model.body_gravcomp[self.body[name]] = 1.0
        for g in self.geoms_of[name]:
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0

    def _enable(self, name: str) -> None:
        self.model.body_gravcomp[self.body[name]] = 0.0
        for g in self.geoms_of[name]:
            self.model.geom_contype[g] = self.geom_type0[g]
            self.model.geom_conaffinity[g] = self.geom_aff0[g]

    def _set_free(self, name: str, pos, quat=(1, 0, 0, 0)) -> None:
        qa, da = self.free_qadr[name], self.free_dadr[name]
        self.data.qpos[qa: qa + 3] = pos
        self.data.qpos[qa + 3: qa + 7] = quat
        self.data.qvel[da: da + 6] = 0.0

    def _configure_rest(self, kind: str, count: int, xy: np.ndarray, active: bool) -> None:
        """Show `count` cups in the rest-of-stack: bands for all but the last, a solid frustum for the last."""
        m = self.model
        self._enable(f"{kind}_rest")
        n = int(np.clip(count, 0, scene.STACK_COUNT_MAX))
        for i in range(scene.STACK_COUNT_MAX):
            shown = i < n
            solid_last = shown and i == n - 1
            for g in self.vis_geoms[kind][i]:
                rgba = self.geom_rgba0[g].copy()
                if not shown:
                    rgba[3] = 0.0
                m.geom_rgba[g] = rgba
            band, full = self.band_geom[kind][i], self.full_geom[kind][i]
            on_band = shown and not solid_last
            m.geom_contype[band] = self.geom_type0[band] if on_band else 0
            m.geom_conaffinity[band] = self.geom_aff0[band] if on_band else 0
            m.geom_contype[full] = self.geom_type0[full] if solid_last else 0
            m.geom_conaffinity[full] = self.geom_aff0[full] if solid_last else 0
        bid = self.body[f"{kind}_rest"]
        mass = max(0.012, (L.PAPER_MASS if kind == 'paper' else L.PLASTIC_MASS) * n)
        h = L.CUP_HEIGHT + max(n - 1, 0) * L.NEST_STEP
        m.body_mass[bid] = mass
        m.body_ipos[bid] = [0.0, 0.0, 0.5 * h]
        ixx = mass * (3 * 0.032 ** 2 + h ** 2) / 12.0
        m.body_inertia[bid] = [ixx, ixx, 0.5 * mass * 0.032 ** 2]
        self._set_free(f"{kind}_rest", [xy[0], xy[1], L.MAT_SURFACE_Z + 0.0004])

    def reset(self, level: int, material: str = "paper", rand: Optional[Randomization] = None, seed: Optional[int] = None) -> None:
        m, d = self.model, self.data
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._noise_rng = np.random.default_rng(seed + 17)
        self.level = int(level)
        self.material = "plastic" if material == "plastic" else "paper"
        if rand is None:
            rand = Randomization()
        self.rand = rand
        self.sensor_noise = rand.sensor_noise
        lay = L.level_layout(self.level, self.material)
        self.layout = lay
        kind = self.material
        other = "plastic" if kind == "paper" else "paper"
        self.active_cup = f"{kind}_cup"
        self.active_rest = f"{kind}_rest"

        mujoco.mj_resetData(m, d)
        for arm in ARMS:
            self.cmd_queue[arm].clear()
        d.xfrc_applied[:] = 0.0

        # randomization of the physical plant (the controller's nominal model is untouched)
        self._restore_nominal_parameters()
        fs = rand.friction_scale
        for g in range(m.ngeom):
            if m.geom_contype[g] and m.geom_friction[g, 0] > 0:
                m.geom_friction[g, 0] = self.geom_friction0[g] * fs
        for arm in ARMS:
            self.grav_adapt[arm] = np.ones(6)
            self.kp[arm] = L.KP_ARM * rand.kp_scale
            self.kd[arm] = L.KD_ARM * rand.kd_scale
            mount = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_mount"))
            base, yaw = L.MOUNTS[arm]
            m.body_pos[mount] = base + rand.base_offset[arm]
            yw = yaw + rand.base_yaw[arm]
            m.body_quat[mount] = [np.cos(0.5 * yw), 0, 0, np.sin(0.5 * yw)]

        jit = rand.object_jitter_xy
        def j(xy):
            if jit <= 0:
                return np.asarray(xy, dtype=float)
            return np.asarray(xy, dtype=float) + self.rng.uniform(-jit, jit, 2)

        # fixed furniture
        m.body_pos[self.body["coaster"]] = [*j(lay.coaster_xy), L.MAT_SURFACE_Z]
        self.dispenser_nominal = L.DISPENSER_XY.copy()
        d.qpos[self.yaw_q] = rand.lever_yaw if rand.lever_yaw is not None else L.LEVER_YAW_DEFAULT
        d.qpos[self.lever_q] = 0.0

        # cups and stacks
        for name in ("paper_cup", "plastic_cup", "paper_rest", "plastic_rest"):
            self._park(name)
        self.tether = {"on": False}
        mass_scale = rand.cup_mass_scale
        for nm_, base in (("paper_cup", L.PAPER_MASS), ("plastic_cup", L.PLASTIC_MASS)):
            bid = self.body[nm_]
            m.body_mass[bid] = base * mass_scale
            m.body_inertia[bid] = self.cup_inertia0[nm_] * mass_scale
        decor_count = 7
        if lay.inverted_stack:
            n = int(rand.stack_count if rand.stack_count is not None else L.NEST_DEFAULT)
            n = int(np.clip(n, 2, L.NEST_MAX))
            self.stack_count = n
            xy = j(lay.cup_xy)
            self._configure_rest(kind, n - 1, xy, True)
            self._configure_rest(other, decor_count, lay.paper_stack_xy if other == "paper" else lay.plastic_stack_xy, False)
            self._enable(self.active_cup)
            top_z = L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP + 0.5 * L.CUP_HEIGHT
            self._set_free(self.active_cup, [xy[0], xy[1], top_z], quat=(0, 1, 0, 0))
            self._arm_tether(kind, break_scale=rand.stack_break_scale)
        else:
            self.stack_count = decor_count + 1
            self._configure_rest("paper", decor_count, lay.paper_stack_xy, False)
            self._configure_rest("plastic", decor_count, lay.plastic_stack_xy, False)
            self._enable(self.active_cup)
            cxy = rand.cup_xy if (rand.cup_xy is not None and level == 5) else j(lay.cup_xy)
            if lay.cup_in_coaster:
                cxy = m.body_pos[self.body["coaster"]][:2].copy()
                z = L.COASTER_FLOOR_Z + 0.5 * L.CUP_HEIGHT + 0.0006
            else:
                z = L.MAT_SURFACE_Z + 0.5 * L.CUP_HEIGHT + 0.0006
            yaw = float(self.rng.uniform(-np.pi, np.pi))
            self._set_free(self.active_cup, [cxy[0], cxy[1], z], quat=(np.cos(0.5 * yaw), 0, 0, np.sin(0.5 * yaw)))
        mujoco.mj_setConst(m, self.scratch)

        # arms at rest, grippers open
        for arm in ARMS:
            a = self.arm[arm]
            d.qpos[a.qpos] = L.REST_Q
            d.qpos[a.g_qpos] = 0.5 * L.GRIPPER_MAX_OPEN_M
            c = self.ctl[arm] = self._new_ctl()
            c["q_cmd"] = L.REST_Q.copy()
            self.set_grip_cap(arm, L.FORCE_CAP_PAPER_CUP_N)
        mujoco.mj_forward(m, d)
        self._arm_tether_relpose()
        self._gravity_nominal()
        self._grav_tick = 0
        self.flow_time = 0.0
        self.flow_time_with_cup = 0.0
        self.flow_events = []
        self.violations = {}
        self._flow_on = False
        self.peak_pad_force = 0.0
        self.history.clear()
        self.dispenser_rest = np.array(d.xpos[self.body["dispenser"]][:2])
        self.settle(0.35)
        self.dispenser_rest = np.array(d.xpos[self.body["dispenser"]][:2])
        if self.tether.get("on"):
            self.tether.update(path=0.0, dir=0, ext=0.0, z0=0.0, last_turn=0.0)

    def _restore_nominal_parameters(self) -> None:
        if not hasattr(self, "geom_friction0"):
            self.geom_friction0 = self.model.geom_friction[:, 0].copy()
            self.cup_inertia0 = {n: self.model.body_inertia[self.body[n]].copy() for n in ("paper_cup", "plastic_cup")}
        self.model.geom_friction[:, 0] = self.geom_friction0

    def held_cup_hits_dispenser(self, q: np.ndarray, grip: float) -> bool:
        """Would an upright cup, centred on this hand pose, touch the valve, spout or tank?

        The arm-only collision check ignores the cup, so a carry can be clear for the links and still
        knock the valve with the cup wall. The cup is placed from the hand pose, not from the true cup."""
        m = self.model
        saved = [(g, int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in self.geoms_of[self.active_cup]]
        for g, _, _ in saved:
            m.geom_contype[g] = self.geom_type0[g]
            m.geom_conaffinity[g] = self.geom_aff0[g]
        try:
            s = self.scratch
            s.qpos[:] = self.data.qpos
            a = self.arm["cup"]
            s.qpos[a.qpos] = q
            s.qpos[a.g_qpos] = 0.5 * float(grip)
            mujoco.mj_kinematics(m, s)
            R = np.array(s.site_xmat[a.site]).reshape(3, 3)
            if float(R[2, 0]) < 0.75:
                return False
            p = np.array(s.site_xpos[a.site])
            qa = self.free_qadr[self.active_cup]
            s.qpos[qa: qa + 7] = [float(p[0]), float(p[1]), float(p[2]), 1.0, 0.0, 0.0, 0.0]
            mujoco.mj_kinematics(m, s)
            mujoco.mj_collision(m, s)
            cup_g = set(self.geoms_of[self.active_cup])
            disp = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
                    ("tank_hit", "valve_barrel", "valve_body", "spout_tip", "lever_stem", "lever_paddle")}
            for i in range(s.ncon):
                c = s.contact[i]
                if c.dist > 0.004:
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in cup_g and g2 in disp) or (g2 in cup_g and g1 in disp):
                    return True
            return False
        finally:
            for g, ct, ca in saved:
                m.geom_contype[g] = ct
                m.geom_conaffinity[g] = ca

    def pin_touches_free_cup(self, q: np.ndarray, grip: float) -> bool:
        """True if this tap pose would touch the top cup, not only the cups underneath it.

        The planner otherwise ignores the top cup, so a short stack lets the pin's funnel land on it."""
        m = self.model
        saved = [(g, int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in self.geoms_of[self.active_cup]]
        for g, _, _ in saved:
            m.geom_contype[g] = self.geom_type0[g]
            m.geom_conaffinity[g] = self.geom_aff0[g]
        try:
            s = self.scratch
            s.qpos[:] = self.data.qpos
            a = self.arm["tap"]
            s.qpos[a.qpos] = q
            s.qpos[a.g_qpos] = 0.5 * float(grip)
            mujoco.mj_kinematics(m, s)
            mujoco.mj_collision(m, s)
            cup_g = set(self.geoms_of[self.active_cup])
            arm_g = self._arm_geoms["tap"]
            for i in range(s.ncon):
                c = s.contact[i]
                if c.dist > 0.0:
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in arm_g and g2 in cup_g) or (g2 in arm_g and g1 in cup_g):
                    return True
            return False
        finally:
            for g, ct, ca in saved:
                m.geom_contype[g] = ct
                m.geom_conaffinity[g] = ca

    def _cup_collides_normally(self, name: str) -> None:
        for g in self.geoms_of[name]:
            self.model.geom_contype[g] = self.geom_type0[g]
            self.model.geom_conaffinity[g] = self.geom_aff0[g]

    def _cup_stuck_mode(self, name: str) -> None:
        """While nested, the top cup touches the arms but not the stack it sits in (the bond carries its weight)."""
        for g in self.geoms_of[name]:
            if self.geom_type0[g] != 0:
                self.model.geom_contype[g] = 8
                self.model.geom_conaffinity[g] = 6

    def _arm_tether(self, kind: str, break_scale: float = 1.0) -> None:
        self.tether = dict(on=False, pending=True, k=3000.0, k_lat=1500.0, c=10.0, kr=0.6, cr=0.01, path=0.0, force=0.0,
                           break_n=L.STACK_BREAK_FORCE_N * break_scale, dir=0, ext=0.0, z0=0.0, last_turn=0.0)

    def _arm_tether_relpose(self) -> None:
        tb = self.tether
        if not tb.get("pending"):
            return
        d = self.data
        top, rest = self.body[self.active_cup], self.body[self.active_rest]
        p1, R1 = np.array(d.xpos[top]), np.array(d.xmat[top]).reshape(3, 3)
        p2, R2 = np.array(d.xpos[rest]), np.array(d.xmat[rest]).reshape(3, 3)
        tb["rel_p"] = R2.T @ (p1 - p2)
        tb["rel_R"] = R2.T @ R1
        tb["dir"], tb["ext"], tb["z0"], tb["last_turn"] = 0, 0.0, 0.0, 0.0
        tb["pending"] = False
        tb["on"] = True
        self._cup_stuck_mode(self.active_cup)

    def settle(self, seconds: float) -> None:
        d = self.data
        for arm in ARMS:
            self.ctl[arm]["q_cmd"] = np.array(d.qpos[self.arm[arm].qpos])
        steps = int(seconds / self.dt)
        for _ in range(steps):
            self._apply_controls()
            self._tether_forces()
            mujoco.mj_step(self.model, d)
        d.qvel[:] = 0.0
        d.time = 0.0
