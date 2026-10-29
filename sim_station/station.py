"""The live twin: one simulation thread (real time x speed), one render thread (JPEG frames), one state snapshot.

The browser never touches MuJoCo. It sends commands (reset / run / step / pause / abort / scrub / camera / dance clock)
and receives JPEG frames plus a JSON snapshot. Everything that scores or labels the run lives in `runner.Session`.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import mujoco
import numpy as np
from PIL import Image

from sim_station import audio as A
from sim_station import layout as L
from sim_station.dance import Choreographer, load_grids, save_grid
from sim_station.plant import ARMS
from sim_station.runner import Session
from sim_station.runtime import CTRL_STEPS

SERIES = ("grip_cup", "cap_cup", "grip_tap", "cap_tap", "speed_cup", "speed_tap", "lever", "resid", "resid_thr")
WINDOW_S = 10.0


class Station:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.sess = Session()
        self.plant = self.sess.plant
        self.mode = "IDLE"                    # IDLE RUNNING PAUSED FAULT DONE SCRUB DANCE
        self.speed = 1.0
        self.cfg = dict(level=1, material="paper", seed=0, randomize=False)
        self.cam = dict(az=-28.0, el=-34.0, dist=1.55, lookat=[0.40, 0.0, 0.10], ortho=False, follow=False, overlay=False)
        self.view = dict(w=1280, h=800)
        self.step_times: Dict[str, dict] = {}
        self._step_sig = None
        self._scrub_target = None
        self.telem = {k: deque(maxlen=520) for k in SERIES}
        self.telem_t: deque = deque(maxlen=520)
        self._last_tcp = {a: self.plant.tcp_true(a)[0].copy() for a in ARMS}
        self._tel_tick = 0
        self.frame: Optional[bytes] = None
        self.frame_id = 0
        self.frame_ms = 0.0
        self.sim_rate = 0.0
        self._stop = False
        self._pick_req = None
        self._pick_res = None
        # dance
        self.choreo = Choreographer()
        self.dance = dict(active=False, song="", t=0.0, playing=False, wall=0.0, phase="off", blocked=False, sync=0.0, info={})
        threading.Thread(target=A.analyze_all, daemon=True).start()
        # background evaluation
        self.mc = dict(running=False, progress="", results={}, started=0.0)
        self.proof = dict(running=False, kind="", result=None)
        self._raw = None
        self._raw_ev = threading.Event()
        self._cam_stamp = (0.0, 0.0, 0.0)
        self._last_motion = time.time()
        self._threads = [threading.Thread(target=self._sim_loop, daemon=True), threading.Thread(target=self._render_loop, daemon=True),
                         threading.Thread(target=self._encode_loop, daemon=True)]
        self.reset()
        for t in self._threads:
            t.start()

    # ================================================================== commands
    def reset(self, level: Optional[int] = None, material: Optional[str] = None, seed: Optional[int] = None, randomize: Optional[bool] = None) -> None:
        with self.lock:
            if level is not None:
                self.cfg["level"] = int(level)
            if material is not None:
                self.cfg["material"] = "plastic" if material == "plastic" else "paper"
            if seed is not None:
                self.cfg["seed"] = int(seed)
            if randomize is not None:
                self.cfg["randomize"] = bool(randomize)
            self.dance["active"] = False
            self.dance["phase"] = "off"
            self.sess.reset(self.cfg["level"], self.cfg["material"], seed=self.cfg["seed"], draw=self.cfg["randomize"])
            self.step_times = {}
            for d in self.telem.values():
                d.clear()
            self.telem_t.clear()
            self._last_tcp = {a: self.plant.tcp_true(a)[0].copy() for a in ARMS}
            self.mode = "IDLE"

    def run(self) -> None:
        with self.lock:
            if self.dance["active"]:
                return
            if self.mode in ("IDLE", "PAUSED"):
                self.mode = "RUNNING"
                self._step_sig = None

    def pause(self) -> None:
        with self.lock:
            if self.mode == "RUNNING":
                self.mode = "PAUSED"

    def step_once(self) -> None:
        """Run until the next step starts or finishes, then pause."""
        with self.lock:
            if self.mode in ("IDLE", "PAUSED"):
                self._step_sig = self._signature()
                self.mode = "RUNNING"

    def abort(self) -> None:
        with self.lock:
            if self.mode in ("RUNNING", "PAUSED", "IDLE"):
                self.sess.ex.abort("aborted by operator")
                self.mode = "FAULT"

    def set_speed(self, s: float) -> None:
        with self.lock:
            self.speed = float(np.clip(s, 0.25, 4.0))

    def set_camera(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                if k in self.cam and v is not None:
                    self.cam[k] = v

    def set_view(self, w: int, h: int) -> None:
        k = min(1.0, 1440.0 / max(w, 1), 900.0 / max(h, 1))            # never render more pixels than the eye can use; the canvas scales it up
        self.view = dict(w=int(np.clip(w * k, 320, 1440)) // 2 * 2, h=int(np.clip(h * k, 200, 900)) // 2 * 2)

    def scrub(self, step_id: str) -> bool:
        """Go back to the moment a step started by replaying the same seeded run (deterministic) up to that time."""
        with self.lock:
            rec = self.step_times.get(step_id)
            if rec is None or self.dance["active"]:
                return False
            target = rec["t0"]
            self.sess.reset(self.cfg["level"], self.cfg["material"], seed=self.cfg["seed"], draw=self.cfg["randomize"])
            keep = {k: v for k, v in self.step_times.items()}
            self.step_times = keep
            for d in self.telem.values():
                d.clear()
            self.telem_t.clear()
            self._scrub_target = self.sess.t_zero + target
            self.mode = "SCRUB"
            return True

    def pick(self, rx: float, ry: float) -> Optional[List[float]]:
        self._pick_res = None
        self._pick_req = (float(rx), float(ry))
        t0 = time.time()
        while self._pick_res is None and time.time() - t0 < 0.6:
            time.sleep(0.01)
        return self._pick_res if self._pick_res not in (None, False) else None

    # ---------------------------------------------------------------- dance
    def dance_start(self, song: str) -> None:
        with self.lock:
            self.cfg["level"] = 1
            self.sess.reset(1, self.cfg["material"], seed=self.cfg["seed"], draw=False, calibrate=False)     # no power-up routine: instant
            self.step_times = {}
            self.choreo = Choreographer()
            for a in ARMS:
                self.choreo.q[a] = self.plant.arm_q(a).copy()
            self.dance.update(active=True, song=song, t=0.0, playing=False, wall=time.time(), phase="follow")
            self.mode = "DANCE"

    def dance_clock(self, song: str, t: float, playing: bool, sync_ms: float = 0.0) -> None:
        with self.lock:
            if not self.dance["active"]:
                return
            self.dance.update(song=song, t=float(t), playing=bool(playing), wall=time.time(), sync=float(sync_ms) / 1000.0)

    def dance_stop(self) -> None:
        with self.lock:
            if self.dance["active"]:
                self.dance["phase"] = "leave"

    def dance_grid(self, song: str, bpm: float, downbeat_s: float) -> None:
        save_grid(song, bpm, downbeat_s)

    # ================================================================== simulation thread
    def _signature(self):
        ex = self.sess.ex
        return (tuple(sorted(ex.done_ids())), tuple(sorted(ex.running())))

    def _record_steps(self) -> None:
        ex = self.sess.ex
        t_zero = self.sess.t_zero
        for s in self.sess.steps:
            if s.id in ex.t0 and s.id not in self.step_times:
                self.step_times[s.id] = dict(t0=ex.t0[s.id] - t_zero, t1=None)
            if s.id in ex.t1 and s.id in self.step_times and self.step_times[s.id]["t1"] is None:
                self.step_times[s.id]["t1"] = ex.t1[s.id] - t_zero

    def _tick_once(self) -> None:
        if self.mode == "DANCE":
            self._dance_tick()
            return
        self.sess.tick()
        self._record_steps()
        self._tel_tick += 1
        if self._tel_tick % 12 == 0:
            self._sample()
        ex = self.sess.ex
        if ex.fault:
            self.mode = "FAULT"
        elif ex.finished:
            self.mode = "DONE"
        elif self.mode == "RUNNING" and self._step_sig is not None and self._signature() != self._step_sig:
            self.mode = "PAUSED"
            self._step_sig = None

    def _dance_tick(self) -> None:
        P = self.plant
        d = self.dance
        now = time.time()
        if d["phase"] == "leave":
            targets = {a: (L.REST_Q.copy(), L.GRIPPER_MAX_OPEN_M) for a in ARMS}
            self.choreo.step(P, targets, CTRL_STEPS * P.dt)
            P.step(CTRL_STEPS)
            if all(np.linalg.norm(self.choreo.q[a] - L.REST_Q) < 0.03 for a in ARMS):
                self.dance["active"] = False
                self.dance["phase"] = "off"
                self.sess.reset(self.cfg["level"], self.cfg["material"], seed=self.cfg["seed"], draw=self.cfg["randomize"], calibrate=False)
                self.mode = "IDLE"
            return
        t_est = d["t"] + ((now - d["wall"]) if d["playing"] else 0.0)
        if d["song"] and d["playing"]:
            targets = self.choreo.pose(d["song"], t_est, d["sync"], load_grids().get(d["song"]))
        else:                                               # no music: stand ready, arms up, swaying gently
            targets = {a: (self.choreo.base.copy(), 0.06) for a in ARMS}
        self.choreo.step(P, targets, CTRL_STEPS * P.dt)
        d["blocked"] = self.choreo.blocked
        d["info"] = dict(self.choreo.info)
        P.step(CTRL_STEPS)
        self._tel_tick += 1
        if self._tel_tick % 12 == 0:
            self._sample()

    def _sample(self) -> None:
        P, ctx = self.plant, self.sess.ctx
        t = P.time - self.sess.t_zero
        dt = 12 * CTRL_STEPS * P.dt
        for arm in ARMS:
            p = P.tcp_true(arm)[0]
            self.telem[f"speed_{arm}"].append(float(np.linalg.norm(p - self._last_tcp[arm]) / dt))
            self._last_tcp[arm] = p.copy()
            self.telem[f"grip_{arm}"].append(float(P.pad_force(arm)))
            self.telem[f"cap_{arm}"].append(float(P.ctl[arm]["grip_cap"]))
        self.telem["lever"].append(float(np.degrees(P.lever_angle())))
        gs = ctx.guard_state
        if gs.get("label"):
            self.telem["resid"].append(float(gs["value"]))
            self.telem["resid_thr"].append(float(gs["threshold"]))
        else:
            mon = [ctx.mon[a].force for a in ARMS if ctx.mon_on[a]]
            self.telem["resid"].append(float(np.linalg.norm(mon[0])) if mon else 0.0)
            self.telem["resid_thr"].append(0.0)
        self.telem_t.append(float(t))

    def _sim_loop(self) -> None:
        owed = 0.0
        last = time.perf_counter()
        meas_t, meas_sim = last, self.plant.time
        while not self._stop:
            now = time.perf_counter()
            dt_wall, last = now - last, now
            with self.lock:
                if self.mode in ("RUNNING", "DANCE"):
                    owed = min(owed + dt_wall * self.speed / (CTRL_STEPS * self.plant.dt), 400.0)
                    n = int(owed)
                    owed -= n
                    for _ in range(n):
                        self._tick_once()
                        if self.mode not in ("RUNNING", "DANCE"):
                            break
                elif self.mode == "SCRUB":
                    t_end = time.perf_counter() + 0.03
                    while time.perf_counter() < t_end:
                        self._tick_once()
                        if self.plant.time >= self._scrub_target or self.mode != "SCRUB":
                            break
                    if self.plant.time >= self._scrub_target:
                        self.mode = "PAUSED"
                        self._step_sig = None
                    elif self.mode == "FAULT":
                        pass
                else:
                    owed = 0.0
            if now - meas_t > 1.0:
                self.sim_rate = (self.plant.time - meas_sim) / (now - meas_t)
                meas_t, meas_sim = now, self.plant.time
            time.sleep(0.003 if self.mode != "SCRUB" else 0.001)

    # ================================================================== render thread
    def _render_loop(self) -> None:
        P = self.plant
        model = P.model
        rdata = mujoco.MjData(model)
        renderer, size = None, None
        smooth_look = np.array(self.cam["lookat"], dtype=float)
        while not self._stop:
            t0 = time.perf_counter()
            w, h = self.view["w"], self.view["h"]
            try:
                if size != (w, h):
                    if renderer is not None:
                        renderer.close()
                    renderer = mujoco.Renderer(model, h, w)
                    size = (w, h)
                    warm = mujoco.MjvCamera()
                    renderer.update_scene(rdata, camera=warm)
                    renderer.render()                      # the first render on a fresh GL context comes out blank
                with self.lock:
                    mujoco.mj_copyData(rdata, model, P.data)
                    cam = dict(self.cam)
                    overlay = self._overlay_markers() if cam["overlay"] else []
                    follow_arm = "tap" if (self.sess.ex.lane.get("tap") and not self.sess.ex.lane.get("cup")) else "cup"
                    look_target = np.array(P.tcp_true(follow_arm)[0]) if cam["follow"] else np.array(cam["lookat"], dtype=float)
                smooth_look += (look_target - smooth_look) * (0.18 if cam["follow"] else 1.0)
                mc = mujoco.MjvCamera()
                mc.type = mujoco.mjtCamera.mjCAMERA_FREE
                mc.azimuth, mc.elevation, mc.distance = float(cam["az"]), float(cam["el"]), float(cam["dist"])
                mc.lookat[:] = smooth_look
                mc.orthographic = 1 if cam["ortho"] else 0
                renderer.update_scene(rdata, camera=mc)
                scn = renderer.scene
                for kind, size_v, pos, rgba, mat in overlay:
                    if scn.ngeom >= scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.array(size_v, dtype=float), np.array(pos, dtype=float), np.array(mat, dtype=float).reshape(-1), np.array(rgba, dtype=np.float32))
                    scn.ngeom += 1
                if self._pick_req is not None:
                    rx, ry = self._pick_req
                    self._pick_req = None
                    pnt = np.zeros(3)
                    gid = np.array([-1], dtype=np.int32)
                    fid = np.array([-1], dtype=np.int32)
                    sid = np.array([-1], dtype=np.int32)
                    body = mujoco.mjv_select(model, rdata, mujoco.MjvOption(), w / h, rx, 1.0 - ry, scn, pnt, gid, fid, sid)
                    self._pick_res = [float(v) for v in pnt] if body >= 0 else False
                self._raw = renderer.render().copy()                # hand the pixels to the encoder thread and go on to the next frame
                self._raw_ev.set()
            except Exception as exc:                      # a render hiccup must never kill the simulation
                self.frame_ms = -1.0
                time.sleep(0.2)
                print("render:", type(exc).__name__, exc, flush=True)
                size = None
                continue
            took = time.perf_counter() - t0
            self.frame_ms = 1000.0 * took
            stamp = (cam["az"], cam["el"], cam["dist"])
            if self.mode in ("RUNNING", "DANCE", "SCRUB") or stamp != self._cam_stamp or cam["follow"]:
                self._last_motion = time.time()
            self._cam_stamp = stamp
            idle = time.time() - self._last_motion > 1.5                      # nothing is moving: 8 frames a second is plenty
            time.sleep(max(0.0, (1.0 / 8.0 if idle else 1.0 / 50.0) - took))

    def _encode_loop(self) -> None:
        while not self._stop:
            if not self._raw_ev.wait(0.5):
                continue
            self._raw_ev.clear()
            img = self._raw
            if img is None:
                continue
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=72, subsampling=2)
            self.frame = buf.getvalue()
            self.frame_id += 1

    def _overlay_markers(self):
        """Believed (cyan) against true (orange) positions of the cup and the handle, drawn on the mat."""
        P, ctx = self.plant, self.sess.ctx
        eye = np.eye(3)
        out = []
        z = L.MAT_SURFACE_Z + 0.004
        truth = P.cup_state()["xyz"]
        bel = ctx.bel.cup_xy
        if bel is not None:
            out.append((mujoco.mjtGeom.mjGEOM_CYLINDER, [0.044, 0.0015, 0], [bel[0], bel[1], z], [0.1, 0.85, 0.95, 0.55], eye))
        out.append((mujoco.mjtGeom.mjGEOM_CYLINDER, [0.036, 0.0015, 0], [truth[0], truth[1], z + 0.002], [1.0, 0.55, 0.1, 0.6], eye))
        piv = ctx.bel.lever_pivot
        psi = ctx.bel.lever_yaw
        if np.isfinite(psi):
            out.append((mujoco.mjtGeom.mjGEOM_CAPSULE, [0.003, 0.03, 0], [piv[0] + 0.03 * np.cos(psi), piv[1] + 0.03 * np.sin(psi), piv[2] + 0.012],
                        [0.1, 0.85, 0.95, 0.9], np.array([[-np.sin(psi), 0, np.cos(psi)], [np.cos(psi), 0, np.sin(psi)], [0, 1, 0]]).T))
        yaw = P.lever_yaw()
        out.append((mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0025, 0.03, 0], [piv[0] + 0.03 * np.cos(yaw), piv[1] + 0.03 * np.sin(yaw), piv[2] + 0.018],
                    [1.0, 0.55, 0.1, 0.9], np.array([[-np.sin(yaw), 0, np.cos(yaw)], [np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0]]).T))
        return out

    # ================================================================== state snapshot
    def snapshot(self) -> dict:
        with self.lock:
            P, ex, ctx = self.plant, self.sess.ex, self.sess.ctx
            t_zero = self.sess.t_zero
            steps = []
            for s in self.sess.steps:
                t0 = ex.t0.get(s.id)
                t1 = ex.t1.get(s.id)
                dur = (t1 - t0) if (t0 is not None and t1 is not None) else ((P.time - t0) if t0 is not None and ex.status[s.id] == "running" else None)
                steps.append(dict(id=s.id, title=s.title, group=s.group, arm=s.arm, deps=list(s.deps), status=ex.status[s.id], dur=dur,
                                  t0=(t0 - t_zero) if t0 is not None else None, check=s.check, msg=ex.msg[s.id]))
            n = len(self.telem_t)
            tel = {k: [round(float(x), 3) for x in list(v)[-n:]] for k, v in self.telem.items()}
            tel["t"] = [round(float(x), 3) for x in self.telem_t]
            lever = float(np.degrees(P.lever_angle()))
            res = None
            if self.mode in ("DONE", "FAULT"):
                res = self.sess.check_pass()
                res.pop("violations", None)
            return dict(
                mode=self.mode, cfg=dict(self.cfg), speed=self.speed, t=float(P.time - t_zero), sim_rate=round(self.sim_rate, 2),
                frame_id=self.frame_id, frame_ms=round(self.frame_ms, 1), progress=ex.progress(), fault=ex.fault,
                steps=steps, events=[(round(t - t_zero, 2), m) for t, m in list(ctx.events)[-60:]], telemetry=tel,
                pour=dict(lever_deg=round(lever, 1), flow_s=round(P.flow_time_with_cup, 2), target_s=7.0, flowing=bool(lever > np.degrees(L.LEVER_FLOW_THRESHOLD))),
                result=res, cam=dict(self.cam), dance=dict(self.dance), mc=dict(self.mc), proof=dict(self.proof),
                tracks=[dict(id=k, title=v["title"], artist=v["artist"], url=f"/audio/{k}", bpm=(round(A.load(k).bpm, 1) if A.load(k) else None),
                             beats=(len(A.load(k).beats) if A.load(k) else 0), duration=(round(A.load(k).duration) if A.load(k) else 0)) for k, v in A.SONGS.items()],
                analysis=dict(A.status),
                beliefs=dict(cup=[round(float(x), 3) for x in ctx.bel.cup_xy] if ctx.bel.cup_xy is not None else None,
                             lever_yaw=(round(float(np.degrees(ctx.bel.lever_yaw)), 1) if np.isfinite(ctx.bel.lever_yaw) else None),
                             lever_yaw_known=bool(np.isfinite(ctx.bel.lever_yaw)),
                             true_lever_yaw=round(float(np.degrees(P.lever_yaw())), 1), true_cup=[round(float(x), 3) for x in P.cup_state()["xyz"][:2]]),
                violations=[(v["kind"], v["a"], v["b"]) for v in P.violations.values() if v["kind"] != "arm_mat"][:6],
            )

    # ================================================================== background proofs
    def start_montecarlo(self, levels: List[int], trials: int, jobs: int = 6) -> None:
        if self.mc["running"]:
            return
        from sim_station import evaluate

        def work() -> None:
            self.mc.update(running=True, progress="starting", results={}, started=time.time())
            try:
                for lv in levels:
                    def prog(level, i, n, r):
                        self.mc["progress"] = f"Level {level}: {i}/{n} trials"
                    out = evaluate.run_level(lv, trials, jobs, progress=prog)
                    out.pop("results", None)
                    self.mc["results"][str(lv)] = out
            finally:
                self.mc.update(running=False, progress="done")

        threading.Thread(target=work, daemon=True).start()

    def start_proof(self, kind: str) -> None:
        if self.proof["running"]:
            return
        from sim_station import proofs

        def work() -> None:
            self.proof.update(running=True, kind=kind, result=None)
            try:
                res = proofs.replay_l1() if kind == "replay" else proofs.equivalence_l1()
            except Exception as exc:
                res = dict(error=f"{type(exc).__name__}: {exc}")
            self.proof.update(running=False, result=res)

        threading.Thread(target=work, daemon=True).start()

    def close(self) -> None:
        self._stop = True
