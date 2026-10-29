"""Cooperative runtime. Skills are generators; each `yield` is one 2 ms control tick for one arm lane.

`Ctx` gives skills honest sensing (encoders, motor torque, gripper width/effort) and motion primitives.
`Executor` runs a dependency graph of steps on two arm lanes plus a global lane, so both arms move at the
same time wherever the plan allows it.
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Generator, List, Optional, Tuple

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.kin import ArmKin, frame_from, rot_about
from sim_station.motion import PlanError, Traj
from sim_station.plant import ARMS, Plant
from sim_station.sensing import Baseline, ContactMonitor

CTRL_STEPS = 2


class SkillFault(RuntimeError):
    """Raised by a skill when a sensed check fails. The message is shown to the operator."""


@dataclass
class Beliefs:
    level: int
    material: str
    coaster_xy: np.ndarray
    spout_xy: np.ndarray
    lever_pivot: np.ndarray
    lever_yaw: float
    cup_xy: Optional[np.ndarray]
    stack_xy: Optional[np.ndarray]
    stack_n: Optional[int]
    stack_top_z: Optional[float] = None
    cup_in_hand: bool = False
    hand_tcp_above_base: float = L.CUP_GRASP_HEIGHT_ABOVE_BASE
    lever_open_flow_s: float = 0.0
    sideways_cup: Optional[dict] = None
    flip_result: Optional[dict] = None
    found: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def from_layout(level: int, material: str) -> "Beliefs":
        lay = L.level_layout(level, material)
        stack_xy = lay.cup_xy if lay.inverted_stack else None
        return Beliefs(
            level=level, material=material,
            coaster_xy=lay.coaster_xy.copy(), spout_xy=L.MOUTH_WORLD_XY.copy(), lever_pivot=L.LEVER_PIVOT_WORLD.copy(),
            lever_yaw=DEFAULT_LEVER_YAW if lay.lever_yaw_known else float("nan"),
            cup_xy=lay.cup_xy.copy() if (lay.cup_location_known and not lay.inverted_stack) else None,
            stack_xy=None if stack_xy is None else stack_xy.copy(),
            stack_n=L.NEST_DEFAULT if (lay.inverted_stack and lay.stack_height_known) else None,
        )


DEFAULT_LEVER_YAW = L.LEVER_YAW_DEFAULT


class Ctx:
    def __init__(self, plant: Plant, kin: Optional[Dict[str, ArmKin]] = None) -> None:
        self.plant = plant
        self.kin = kin or {a: ArmKin(a) for a in ARMS}
        self.mon = {a: ContactMonitor(a, plant, self.kin[a]) for a in ARMS}
        self.mon_on = {a: False for a in ARMS}
        self.q_cmd = {a: plant.arm_q(a).copy() for a in ARMS}
        self.bel = Beliefs.from_layout(plant.level, plant.material)
        self.events: Deque[Tuple[float, str]] = deque(maxlen=300)
        self.speed = 1.0
        self.dt = L.CONTROL_DT
        self.pour_s = 7.0
        self.flow_belief = 0.0
        self.trace: List[dict] = []
        self.scratch: Dict[str, object] = {}
        self.guard_state: Dict[str, object] = {"value": 0.0, "threshold": 0.0, "label": ""}
        self.lever_held = False
        self.flow_clock_on = False
        self.sense(force=True)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.bel = Beliefs.from_layout(self.plant.level, self.plant.material)
        self.q_cmd = {a: self.plant.arm_q(a).copy() for a in ARMS}
        self.flow_belief = 0.0
        self.scratch = {}
        self.guard_state = {"value": 0.0, "threshold": 0.0, "label": ""}
        self.lever_held = False
        self.flow_clock_on = False
        self.sense(force=True)

    def event(self, msg: str) -> None:
        self.events.append((self.plant.time, msg))

    def sense(self, force: bool = False) -> None:
        for arm in ARMS:
            obs = self.plant.read_arm(arm)
            m = self.mon[arm]
            if self.mon_on[arm] or force:
                m.update(obs, settle=force)
            else:
                m.obs = obs
                m.q, m.qd = obs.q, obs.qd
                m.grip_width, m.grip_vel, m.grip_force = obs.grip_width, obs.grip_vel, obs.grip_force
        if self.flow_clock_on and self.lever_held and self.mon_on["tap"]:
            if self.lever_angle_belief() > L.LEVER_FLOW_THRESHOLD:
                self.flow_belief += CTRL_STEPS * self.plant.dt

    def lever_angle_belief(self) -> float:
        m = self.mon["tap"]
        r = m.tcp[:3, 3] - self.bel.lever_pivot
        psi = self.bel.lever_yaw
        d = np.array([np.cos(psi), np.sin(psi), 0.0])
        return float(np.arctan2(r[2], float(r @ d)) - np.arctan2(0.0015 + 0.003, L.LEVER_GRIP_DIST))

    def monitor(self, arm: str, on: bool) -> None:
        if on and not self.mon_on[arm]:
            self.mon_on[arm] = True
            self.mon[arm].update(self.plant.read_arm(arm), settle=True)
        elif not on:
            self.mon_on[arm] = False

    def q(self, arm: str) -> np.ndarray:
        return self.mon[arm].q.copy()

    def tcp(self, arm: str) -> np.ndarray:
        return self.kin[arm].fk(self.mon[arm].q)

    def width(self, arm: str) -> float:
        return self.mon[arm].grip_width

    # ------------------------------------------------------- generator helpers
    def wait(self, seconds: float):
        n = max(int(round(seconds / (CTRL_STEPS * self.plant.dt))), 1)
        for _ in range(n):
            self.hold(None)
            yield

    def settle(self, arm: str, tol: float = 0.0015, timeout: float = 3.0):
        """Keep the last command until the hand has caught up with it: a load (a pushed handle, a pressed stack) leaves a
        position-controlled arm lagging by force/stiffness, and it only arrives once the load gives way or the force balances."""
        T_cmd = self.kin[arm].fk(self.q_cmd[arm])[:3, 3]
        n = max(int(round(timeout / (CTRL_STEPS * self.plant.dt))), 1)
        calm = 0
        for _ in range(n):
            self.hold(arm)
            yield
            err = float(np.linalg.norm(self.tcp(arm)[:3, 3] - T_cmd))
            calm = calm + 1 if err < tol else 0
            if calm >= 12:
                return True
        return False

    def hold(self, arm: Optional[str]) -> None:
        for a in ARMS if arm is None else (arm,):
            self.plant.command_arm(a, self.q_cmd[a], np.zeros(6))

    def play(self, arm: str, traj: Traj, guard: Optional[Callable[[], bool]] = None, on_tick: Optional[Callable[[int], None]] = None):
        """Follow a trajectory. Returns True if it completed, False if the guard fired."""
        for k in range(len(traj)):
            if guard is not None and guard():
                self.q_cmd[arm] = self.mon[arm].q.copy()
                self.plant.command_arm(arm, self.q_cmd[arm], np.zeros(6))
                return False
            self.q_cmd[arm] = traj.q[k]
            self.plant.command_arm(arm, traj.q[k], traj.qd[k])
            if on_tick is not None:
                on_tick(k)
            yield
        return True

    def check_traj(self, arm: str, traj: Traj, other_q: Optional[np.ndarray] = None, stride_s: float = 0.03,
                   grip_width: Optional[float] = None) -> None:
        """Planner-level collision test of the swept path against the world and the other arm."""
        other = "tap" if arm == "cup" else "cup"
        qo = self.q_cmd[other] if other_q is None else other_q
        w = self.width(arm) if grip_width is None else grip_width
        step = max(int(stride_s / traj.dt), 1)
        k0 = min(int(0.12 / traj.dt), len(traj) - 1)         # a path may start touching what the last guarded move touched
        # contacts already present where the path begins are what we are leaving, not what we would run into
        leaving = {tuple(sorted(h)) for h in self.plant.collisions_for({arm: traj.q[0], other: qo}, {arm: w, other: self.width(other)})}
        for k in list(range(k0, len(traj), step)) + [len(traj) - 1]:
            hits = self.plant.collisions_for({arm: traj.q[k], other: qo}, {arm: w, other: self.width(other)})
            hits = [h for h in hits if not self._benign(h) and tuple(sorted(h)) not in leaving]
            if hits:
                raise PlanError(f"{arm} arm path collides at {k * traj.dt:.2f}s: {hits[0][0]} / {hits[0][1]}")
            if arm == "cup" and self.bel.cup_in_hand and self.plant.held_cup_hits_dispenser(traj.q[k], w):
                raise PlanError(f"the held cup would bump the valve or the tank at {k * traj.dt:.2f}s")

    @staticmethod
    def _benign(pair: Tuple[str, str]) -> bool:
        a, b = pair
        return False

    def cart_repair(self, arm: str, poses: List[np.ndarray], v_cart: float, min_T: float = 0.8, **kw):
        """cart_move that falls back to re-seating the wrist on a known-good elbow/wrist branch when the straight line runs
        into a joint limit on the branch the arm happens to be on. Returns the trajectories to play, in order."""
        kin = self.kin[arm]
        try:
            return [M.cart_move(kin, poses, self.q(arm), v_cart=v_cart, speed_scale=self.speed, min_T=min_T, **kw)]
        except PlanError as first:
            seeds = [L.REAL_L1_CUP_WAYPOINTS["in_coaster"][:6], L.REAL_L1_CUP_WAYPOINTS["paper_hold"][:6],
                     L.REAL_L1_CUP_WAYPOINTS["under_spout"][:6], np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0])] if arm == "cup" else \
                [np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.5, 1.2, 0.8, 0.6, 0.0, 0.0])]
            best = None
            for seed in seeds:
                end = kin.ik(poses[-1], seed)
                start = kin.ik(poses[0], seed)
                if end.ok and start.ok:
                    best = start
                    break
            if best is None:
                raise first
            hop = M.joint_move(self.q(arm), best.q, speed_scale=self.speed, min_T=0.8)
            self.check_traj(arm, hop)
            line = M.cart_move(kin, poses, best.q, v_cart=v_cart, speed_scale=self.speed, min_T=min_T, **kw)
            self.event(f"re-seated the {arm} wrist on a reachable branch before the straight move")
            return [hop, line]

    # ----------------------------------------------------------------- gripper
    def grip_open(self, arm: str, width: float = L.GRIPPER_MAX_OPEN_M, cap: float = L.FORCE_CAP_PROBE_N, rate: float = L.GRIPPER_RATE_M_S,
                  tol: float = 0.0012, timeout: float = 2.5):
        self.plant.command_gripper(arm, width, cap_n=cap, rate=rate)
        t0 = self.plant.time
        while abs(self.width(arm) - width) > tol and self.plant.time - t0 < timeout:
            self.hold(arm)
            yield
        return self.width(arm)

    def grip_close(self, arm: str, cap: float, rate: float = 0.03, target: float = 0.0, timeout: float = 4.0, min_width: float = 0.003):
        """Close with a force cap until the jaws stall. Returns dict(width, force, stalled)."""
        self.plant.command_gripper(arm, target, cap_n=cap, rate=rate)
        t0 = self.plant.time
        still = 0.0
        last_w = self.width(arm)
        while self.plant.time - t0 < timeout:
            self.hold(arm)
            yield
            m = self.mon[arm]
            moving = abs(m.grip_vel) > 0.004
            still = 0.0 if moving else still + self.dt * CTRL_STEPS * 0.5
            if m.grip_width < min_width * 0.5:
                break
            if still > 0.16 and m.grip_force > 0.6 * cap and m.grip_width > min_width:
                break
            if still > 0.5 and m.grip_width > min_width:
                break
        w = self.width(arm)
        stalled = w > min_width
        # park the servo reference just past the contact so it holds the cap force without winding up
        self.plant.ctl[arm]["grip_target"] = max(w - 0.0015, 0.0)
        self.plant.ctl[arm]["grip_ref"] = max(w - 0.0015, 0.0)
        return dict(width=w, force=self.mon[arm].grip_force, stalled=stalled)

    # ---------------------------------------------------- frame / pose helpers
    def side_frames(self, arm: str, xy: np.ndarray, z: float, depth: float = 0.004, standoff: float = 0.05,
                    tilts=(0.0, -15.0, -30.0, -45.0), offsets=(-9.0, 0.0, -20.0), lift: float = 0.0,
                    seeds: Optional[List[np.ndarray]] = None, extra_up: float = 0.0, az_fixed: Optional[float] = None):
        """Reachable side-grasp frames: grasp, pre-grasp standoff, high via point, and their joint solutions.

        `az_fixed` pins the approach azimuth (radians) instead of deriving it from the arm base, for grasps whose jaw axis must
        line up with a known uncertain direction."""
        kin = self.kin[arm]
        base = kin.mount[:2]
        q_now = self.q(arm)
        seed_list = seeds or [L.REAL_L1_CUP_WAYPOINTS["paper_hold"][:6], np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0]),
                              np.array([0.3, 1.2, 0.9, 0.5, 0.0, 0.0]), q_now]
        last = None
        if az_fixed is None:
            azimuths = [np.arctan2(xy[1] - base[1], xy[0] - base[0]) + np.radians(off) for off in offsets]
        else:
            azimuths = [az_fixed, az_fixed + np.radians(12.0), az_fixed - np.radians(12.0)]
        for tilt in tilts:
            for az in azimuths:
                t = np.radians(tilt)
                a = np.array([np.cos(az) * np.cos(t), np.sin(az) * np.cos(t), np.sin(t)])
                j = np.array([-np.sin(az), np.cos(az), 0.0])
                tcp = np.array([xy[0], xy[1], z]) + a * depth
                T_g = frame_from(a, j, tcp)
                T_pre = frame_from(a, j, tcp - a * standoff + np.array([0.0, 0.0, 0.012]))
                T_hi = frame_from(a, j, tcp - a * standoff + np.array([0.0, 0.0, 0.085 + extra_up]))
                T_up = frame_from(a, j, tcp + np.array([0.0, 0.0, 0.05 + lift]))
                qs = {}
                ok = True
                for name, T in (("grasp", T_g), ("pre", T_pre), ("hi", T_hi), ("up", T_up)):
                    r = kin.ik_multi(T, seed_list)
                    if not r.ok:
                        ok = False
                        last = f"{name} pose: {r.reason}"
                        break
                    qs[name] = r.q
                if ok:
                    return dict(grasp=T_g, pre=T_pre, hi=T_hi, up=T_up, q=qs, tilt=tilt, az=float(az), a=a, j=j)
        raise PlanError(f"no reachable side grasp at ({xy[0]:.3f}, {xy[1]:.3f}, {z:.3f}): {last}")


@dataclass
class StepDef:
    id: str
    title: str
    group: str
    arm: Optional[str]
    deps: Tuple[str, ...]
    make: Callable[["Ctx"], Generator]
    check: str = ""


class Executor:
    """Runs a step graph. Statuses: waiting, running, done, failed."""

    def __init__(self, ctx: Ctx, steps: List[StepDef]) -> None:
        self.ctx = ctx
        self.steps = steps
        self.by_id = {s.id: s for s in steps}
        self.status: Dict[str, str] = {s.id: "waiting" for s in steps}
        self.gens: Dict[str, Generator] = {}
        self.msg: Dict[str, str] = {s.id: "" for s in steps}
        self.t0: Dict[str, float] = {}
        self.t1: Dict[str, float] = {}
        self.lane: Dict[Optional[str], Optional[str]] = {"cup": None, "tap": None}
        self.fault: str = ""
        self.finished = False
        self.only: Optional[str] = None
        self.snapshots: Dict[str, dict] = {}
        self.started_at = ctx.plant.time

    def done_ids(self) -> set:
        return {i for i, s in self.status.items() if s == "done"}

    def running(self) -> List[str]:
        return [i for i, s in self.status.items() if s == "running"]

    def progress(self) -> float:
        done = sum(1 for s in self.status.values() if s == "done")
        return done / max(len(self.steps), 1)

    def _can_start(self, s: StepDef, done: set) -> bool:
        if self.status[s.id] != "waiting":
            return False
        if self.only is not None and s.id != self.only:
            return False
        if not all(d in done or (self.only is not None) for d in s.deps):
            return False
        if s.arm is not None and self.lane.get(s.arm) is not None:
            return False
        return True

    def start_step(self, s: StepDef) -> None:
        self.status[s.id] = "running"
        self.t0[s.id] = self.ctx.plant.time
        try:
            self.snapshots[s.id] = self.ctx.plant.save_state()
        except Exception:
            pass
        self.gens[s.id] = s.make(self.ctx)
        if s.arm is not None:
            self.lane[s.arm] = s.id
        self.ctx.event(f"start {s.id}: {s.title}")

    def tick(self) -> None:
        ctx = self.ctx
        ctx.sense()
        if not self.fault and not self.finished:
            done = self.done_ids()
            for s in self.steps:
                if self._can_start(s, done):
                    self.start_step(s)
            busy = set()
            for sid in list(self.running()):
                s = self.by_id[sid]
                try:
                    next(self.gens[sid])
                except StopIteration as stop:
                    res = stop.value or {}
                    self.status[sid] = "done"
                    self.t1[sid] = ctx.plant.time
                    self.msg[sid] = str(res.get("msg", "")) if isinstance(res, dict) else str(res)
                    if s.arm is not None:
                        self.lane[s.arm] = None
                    ctx.event(f"done {sid}: {self.msg[sid]}")
                except (SkillFault, PlanError) as exc:
                    self._fail(sid, str(exc))
                except Exception as exc:  # genuine bug: surface it, do not hide
                    self._fail(sid, f"{type(exc).__name__}: {exc}")
                    ctx.event(traceback.format_exc().splitlines()[-3])
            if self.only is not None:
                if self.status.get(self.only) in ("done", "failed"):
                    self.finished = True
            elif all(v == "done" for v in self.status.values()):
                self.finished = True
        for arm in ARMS:
            if self.lane.get(arm) is None:
                ctx.hold(arm)
        ctx.plant.step(CTRL_STEPS)

    def _fail(self, sid: str, why: str) -> None:
        self.status[sid] = "failed"
        self.t1[sid] = self.ctx.plant.time
        self.msg[sid] = why
        self.fault = f"{sid}: {why}"
        s = self.by_id[sid]
        if s.arm is not None:
            self.lane[s.arm] = None
        self.ctx.event(f"FAULT {sid}: {why}")

    def abort(self, why: str = "aborted by operator") -> None:
        if not self.fault:
            self.fault = why
        for sid in self.running():
            self.status[sid] = "failed"
            self.msg[sid] = why
        for arm in ARMS:
            self.lane[arm] = None
            self.ctx.q_cmd[arm] = self.ctx.mon[arm].q.copy()
            self.ctx.hold(arm)
        self.ctx.event(why)

    def run_blocking(self, max_sim_s: float = 400.0, wall_s: float = 600.0) -> None:
        t0 = time.time()
        tmax = self.ctx.plant.time + max_sim_s
        while not self.finished and not self.fault and self.ctx.plant.time < tmax and time.time() - t0 < wall_s:
            self.tick()
