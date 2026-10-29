"""Render the project-page videos headlessly from the live physics.

    python tools/make_videos.py level 2 site/videos/level2.mp4
    python tools/make_videos.py dance rick site/videos/dance_rick.mp4 [start_s] [seconds]

Dance videos carry no audio track (the songs are copyrighted); the dots show the tracked beat. Captions are drawn from the executor's own step titles.
"""
import json
import subprocess
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sim_station
from sim_station import audio as A
from sim_station import layout as L
from sim_station.dance import Choreographer
from sim_station.plant import ARMS
from sim_station.runner import Session
from sim_station.runtime import CTRL_STEPS

W, H, FPS = 960, 540, 15
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except Exception:
    FONT = SMALL = ImageFont.load_default()


class Sink:
    def __init__(self, out):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(img.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def caption(img, lines, right=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, H - 64, W, H], fill=(14, 16, 21, 215))
    d.text((18, H - 56), lines[0], font=FONT, fill=(231, 234, 241))
    if len(lines) > 1:
        d.text((18, H - 28), lines[1], font=SMALL, fill=(138, 147, 168))
    if right:
        d.text((W - 18 - d.textlength(right, font=SMALL), H - 28), right, font=SMALL, fill=(138, 147, 168))
    return im


def level(lv: int, out: str) -> None:
    S = Session()
    S.reset(lv, "paper", seed=0)
    P = S.plant
    r = mujoco.Renderer(P.model, H, W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = -28.0, -34.0, 1.45
    cam.lookat[:] = [0.42, 0.0, 0.08]
    r.update_scene(P.data, camera=cam); r.render()
    sink, nxt = Sink(out), 0.0
    names = {s.id: s.title for s in S.steps}
    groups = {s.id: s.group for s in S.steps}
    chapters = []
    seen = set()
    scalar = ("t", "cup_w", "tap_w", "cup_f", "tap_f", "cup_cap", "tap_cap", "cup_speed", "tap_speed", "contact", "contact_thr", "lever", "flow")
    tel = {k: [] for k in scalar}
    joints = {f"{a}_{k}": [] for a in ARMS for k in ("q", "qd", "tau")}
    tel_nxt, last_flow = 0.0, P.flow_time
    while not S.ex.finished and not S.ex.fault and P.time - S.t_zero < 400:
        S.tick()
        t = P.time - S.t_zero
        for sid in S.ex.running():
            if sid not in seen:
                seen.add(sid)
                chapters.append(dict(t=round(t, 2), id=sid, title=names[sid], group=groups[sid]))
        if t >= tel_nxt:
            tel_nxt += 0.1
            row = P._history_sample()
            gs = S.ctx.guard_state
            if gs.get("label"):
                contact = float(gs["value"])
            else:
                mon = [S.ctx.mon[a].force for a in ARMS if S.ctx.mon_on[a]]
                contact = float(np.linalg.norm(mon[0])) if mon else 0.0
            tel["t"].append(round(t, 2))
            for a in ARMS:
                obs = P.read_arm(a)
                tel[f"{a}_w"].append(round(row[f"{a}_width_mm"], 1))
                tel[f"{a}_f"].append(round(row[f"{a}_grip_f"], 2))
                tel[f"{a}_cap"].append(round(float(P.ctl[a]["grip_cap"]), 2))
                tel[f"{a}_speed"].append(round(float(row[f"{a}_speed"]) * 1000.0, 1))
                joints[f"{a}_q"].append([round(float(np.degrees(x)), 2) for x in obs.q])
                joints[f"{a}_qd"].append([round(float(np.degrees(x)), 2) for x in obs.qd])
                joints[f"{a}_tau"].append([round(float(x), 3) for x in obs.tau])
            tel["contact"].append(round(contact, 2))
            tel["contact_thr"].append(round(float(gs.get("threshold") or 0.0), 2))
            tel["lever"].append(round(float(np.degrees(row["lever"])), 1))
            tel["flow"].append(int(P.flow_time > last_flow))
            last_flow = P.flow_time
        if t >= nxt:
            nxt += 1.0 / FPS
            r.update_scene(P.data, camera=cam)
            run = S.ex.running()
            title = names[run[0]] if run else "..."
            sink.write(np.asarray(caption(r.render(), [title[:78], f"Level {lv}  ·  paper cup  ·  simulation time {t:5.1f} s"], f"{len(S.ex.done_ids())}/{len(S.steps)} steps done")))
    res = S.check_pass()
    for _ in range(FPS * 2):
        r.update_scene(P.data, camera=cam)
        verdict = "all referee checks passed" if res["ok"] else "did not pass: " + ", ".join(res["reasons"][:3])
        sink.write(np.asarray(caption(r.render(), [f"Level {lv} finished: {verdict}"[:80], f"flow {res['flow_s']:.2f} s  ·  cup {res['coaster_offset_mm']:.0f} mm from the coaster centre  ·  {res['sim_time_s']:.0f} s"])))
    sink.close()
    n = len(tel["t"])
    for key, rows in joints.items():
        tel[key] = [[rows[s][j] for s in range(n)] for j in range(6)]
    summary = dict(level=lv, ok=bool(res["ok"]), flow_s=round(float(res["flow_s"]), 2),
                   coaster_offset_mm=round(float(res["coaster_offset_mm"]), 1), sim_time_s=round(float(res["sim_time_s"]), 1),
                   chapters=chapters, telemetry=tel)
    Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"level {lv}: pass={res['ok']} {res['reasons']}")


def dance(song: str, out: str, start: float = 30.0, seconds: float = 36.0) -> None:
    A.analyze_all()
    an = A.load(song)
    S = Session()
    S.reset(1, "paper", seed=0, calibrate=False)
    P = S.plant
    ch = Choreographer()
    for a in ARMS:
        ch.q[a] = P.arm_q(a).copy()
    r = mujoco.Renderer(P.model, H, W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = -20.0, -26.0, 1.5
    cam.lookat[:] = [0.34, 0.0, 0.22]
    r.update_scene(P.data, camera=cam); r.render()
    sink, nxt, t0 = Sink(out), 0.0, P.time
    dt = CTRL_STEPS * P.dt
    while P.time - t0 < seconds + 4.0:
        t = P.time - t0
        t_song = start + max(t - 4.0, 0.0)                       # four seconds to rise into the ready pose, then the song's beat grid
        targets = ch.pose(song, t_song) if t >= 4.0 else {a: (ch.base.copy(), 0.06) for a in ARMS}
        ch.step(P, targets, dt)
        P.step(CTRL_STEPS)
        if t >= nxt:
            nxt += 1.0 / FPS
            r.update_scene(P.data, camera=cam)
            im = caption(r.render(), [f"{A.SONGS[song]['title']}  ·  {A.SONGS[song]['artist']}", f"move: {ch.info.get('move', 'ready')}  ·  {an.bpm:.0f} BPM tracked from the audio"])
            d = ImageDraw.Draw(im, "RGBA")
            b = ch.info.get("beat", 0.0) if t >= 4.0 else -1
            for i in range(4):
                on = int(b) == i
                d.ellipse([W - 40 - (3 - i) * 34 - (8 if on else 0), 26 - (8 if on else 0), W - 40 - (3 - i) * 34 + (8 if on else 0) + 12, 26 + (8 if on else 0) + 12],
                          fill=(124, 140, 255, 255) if (on and i == 0) else ((76, 208, 138, 255) if on else (60, 68, 100, 200)))
            sink.write(np.asarray(im))
    sink.close()
    print("dance", song, "done")


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "level":
        level(int(sys.argv[2]), sys.argv[3])
    else:
        dance(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 30.0, float(sys.argv[5]) if len(sys.argv) > 5 else 36.0)
