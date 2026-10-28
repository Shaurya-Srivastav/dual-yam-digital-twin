"""Dance Mode: choreography as key poses on the beat, driven by the audio player's clock.

Nothing here touches audio. The browser plays the song (official YouTube embed, or a file the user picked that never leaves
their machine) and reports its `currentTime`; the pose is a pure function of that time, so pause and seek simply work.
Every pose goes through the same joint speed / acceleration limits as the task skills and is checked against the arms'
collision model before it is applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from sim_station import audio as A
from sim_station import layout as L

GRID_FILE = Path(__file__).resolve().parent / "dance_grids.json"
LEAD_S = 0.20                      # pose the arms this far ahead of the audio clock: tracker lag (~0.11 s) + streaming and render lag
OPEN_W, SHUT_W = 0.0776, 0.012


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def snap(x: float, hold: float) -> float:
    """0 for the first `hold` of the beat, then an eased move to 1 (staccato)."""
    return 0.0 if x < hold else ease((x - hold) / max(1.0 - hold, 1e-6))


def load_grids() -> Dict[str, dict]:
    if GRID_FILE.exists():
        try:
            return json.loads(GRID_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_grid(song: str, bpm: float, downbeat_s: float) -> None:
    grids = load_grids()
    grids[song] = dict(bpm=float(bpm), downbeat_s=float(downbeat_s))
    GRID_FILE.write_text(json.dumps(grids, indent=1))



def hit(f: float, k: float = 5.0) -> float:
    """Impulse on the beat that decays through it: the body 'drops' on the one."""
    return float(np.exp(-k * f))


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------------------------------------------- moves
# Each move maps the beat counter b (counts from the downbeat) and the intensity a (0.5..1.3) to joint offsets for the lead arm
# and a gripper opening (fraction of fully open). Everything is phase-locked to the beat counter, so the whole body is on the beat.

def _sway(b, a):
    s = np.sin(2 * np.pi * b / 4.0)
    f = b % 1.0
    return np.array([0.36 * s, 0.04 + 0.07 * hit(f), -0.07 * hit(f), 0.07 * hit(f), 0.16 * s, 0.55 * np.sin(2 * np.pi * b / 4.0 + np.pi / 2)]) * a, 0.55


def _pump(b, a, other=False):
    bb = b + (1.0 if other else 0.0)
    p = 0.5 - 0.5 * np.cos(np.pi * bb)                              # one full up-down every two beats
    f = b % 1.0
    return np.array([0.10 * np.sin(np.pi * bb), -0.30 * p + 0.04 * hit(f), 0.26 * p - 0.04 * hit(f), -0.16 * p, 0.0, 0.25 * np.sin(np.pi * bb)]) * a, 0.7


def _wave(b, a, delay=0.0):
    ph = 2 * np.pi * (b - delay) / 4.0
    f = b % 1.0
    return np.array([0.30 * np.sin(ph), 0.05 * np.sin(2 * np.pi * b / 2.0) + 0.04 * hit(f), -0.05 * np.sin(2 * np.pi * b / 2.0), 0.10 * np.sin(ph - 0.7),
                     0.45 * np.sin(ph - 0.9), 0.95 * np.sin(ph - 1.8)]) * a, 0.5 + 0.3 * np.sin(ph)


def _circle(b, a):
    ph = 2 * np.pi * b / 4.0
    return np.array([0.30 * np.sin(ph), 0.20 * np.cos(ph) + 0.03 * hit(b % 1.0), -0.20 * np.cos(ph), 0.12 * np.sin(ph + 1.0), 0.20 * np.cos(ph), 0.4 * np.sin(ph)]) * a, 0.6


POINTS = np.array([[0.45, -0.30, 0.25, -0.20, 0.30, 0.9], [-0.40, 0.25, -0.20, 0.20, -0.30, -0.9], [0.25, -0.35, 0.30, -0.25, 0.10, 0.5], [0.0, 0.05, -0.05, 0.0, 0.0, 0.0]])


def _point(b, a):
    n, f = int(np.floor(b)) % 4, b % 1.0
    k0, k1 = POINTS[(n - 1) % 4], POINTS[n]
    s = smooth((f - 0.05) / 0.35)                                   # arrive on the beat, then hold: the disco point
    return (k0 + (k1 - k0) * s) * a, 0.35 if n in (0, 2) else 0.8


def _clap(b, a):
    n, f = int(np.floor(b)) % 4, b % 1.0
    snare = n in (1, 3)
    close = hit(f, 9.0) if snare else 0.0
    return np.array([0.20 * (1 if snare else 0.4), 0.05 + 0.06 * hit(f), -0.06 * hit(f), 0.06 * hit(f), 0.0, 0.0]) * a, 1.0 - 0.9 * close


STYLE_PLAN = {                       # eight-beat phrases, in the order they come round
    "rick": ["sway", "pump", "sway", "wave", "clap", "pump", "circle", "point"],
    "aha": ["wave", "circle", "sway", "wave", "circle", "sway", "pump", "point"],
    "disco": ["point", "pump", "point", "clap", "sway", "point", "pump", "circle"],
    "free": ["sway", "pump", "wave", "circle"],
}


def move_pose(name: str, b: float, a: float, arm: str):
    """Lead arm is the cup arm. The tap arm mirrors it, or answers it half a beat or two beats later, depending on the move."""
    if arm == "tap":
        if name == "pump":
            return _pump(b, a, other=True)
        if name == "wave":
            d, w = _wave(b, a, delay=0.5)
            return d * np.array([-1, 1, 1, 1, -1, -1]), w
        if name == "point":
            d, w = _point(b + 2.0, a)
            return d * np.array([-1, 1, 1, 1, -1, -1]), w
        d, w = {"sway": _sway, "circle": _circle, "clap": _clap}[name](b, a)
        return d * np.array([-1, 1, 1, 1, -1, -1]), w
    return {"sway": _sway, "pump": lambda b_, a_: _pump(b_, a_), "wave": lambda b_, a_: _wave(b_, a_), "circle": _circle, "point": _point, "clap": _clap}[name](b, a)


class Choreographer:
    def __init__(self) -> None:
        self.base = L.DANCE_BASE_Q.copy()
        self.q = {a: L.REST_Q.copy() for a in ("cup", "tap")}
        self.v = {a: np.zeros(6) for a in ("cup", "tap")}
        self.w = {a: OPEN_W for a in ("cup", "tap")}
        self.blocked = False
        self.info = {"move": "", "beat": 0.0, "bar": 0, "energy": 0.0}

    # ---------------------------------------------------------------- the pose function
    def pose(self, song: str, t_audio: float, sync_s: float = 0.0, custom_grid: Optional[dict] = None) -> Dict[str, Tuple[np.ndarray, float]]:
        an = A.load(song)
        t = t_audio + LEAD_S + sync_s
        if an is not None:
            b = an.beat_pos(t) - an.bar_offset()
            if an.half_time:
                b *= 0.5
            en = 0.5 * an.energy_at(t) + 0.5 * an.low_at(t)
            style = A.SONGS[song]["style"]
        else:                                               # a file of the user's own: a constant grid from the tap-in
            g = custom_grid or {"bpm": 120.0, "downbeat_s": 0.0}
            b = (t - g["downbeat_s"]) * g["bpm"] / 60.0
            en, style = 0.6, "free"
        plan = STYLE_PLAN[style]
        phrase, pos = int(np.floor(b / 8.0)), b - 8.0 * np.floor(b / 8.0)
        cur, prv = plan[phrase % len(plan)], plan[(phrase - 1) % len(plan)]
        x = smooth(pos / 1.0)                               # cross-fade from the previous move over the first beat of the phrase
        amp = 0.55 + 0.75 * en
        out = {}
        for arm in ("cup", "tap"):
            d1, w1 = move_pose(cur, b, amp, arm)
            d0, w0 = move_pose(prv, b, amp, arm)
            d, w = d0 + (d1 - d0) * x, w0 + (w1 - w0) * x
            q = np.clip(self.base + d, L.JOINT_LOW + 0.05, L.JOINT_HIGH - 0.05)
            out[arm] = (q, float(OPEN_W * np.clip(w, 0.08, 1.0)))
        self.info = {"move": cur, "beat": float(b % 4.0), "bar": int(np.floor(b / 4.0)), "energy": float(en)}
        return out

    # --------------------------------------------------------------- limited following
    def step(self, plant, targets: Dict[str, Tuple[np.ndarray, float]], dt: float) -> None:
        """Track each arm's target with the same velocity and acceleration limits as the skills; refuse targets the collision
        model says would hit the other arm or the bench (queried at 50 Hz)."""
        cand = {}
        for arm in ("cup", "tap"):
            q_t, _ = targets[arm]
            q, v = self.q[arm], self.v[arm]
            a = np.clip(300.0 * (q_t - q) - 34.0 * v, -L.MAX_JOINT_ACC * 1.6, L.MAX_JOINT_ACC * 1.6)
            v = np.clip(v + a * dt, -0.9 * L.MAX_JOINT_VEL, 0.9 * L.MAX_JOINT_VEL)
            cand[arm] = (q + v * dt, v)
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 10 == 1:
            hits = plant.collisions_for({a: cand[a][0] for a in cand}, {a: targets[a][1] for a in targets})
            self._hits = bool(hits)
        if getattr(self, "_hits", False):
            self.blocked = True
            for arm in cand:
                self.v[arm] = np.zeros(6)
        else:
            self.blocked = False
            for arm in cand:
                self.q[arm], self.v[arm] = cand[arm]
        for arm in ("cup", "tap"):
            plant.command_arm(arm, self.q[arm], self.v[arm])
            self.w[arm] = targets[arm][1]
            plant.command_gripper(arm, self.w[arm], cap_n=6.0, rate=0.9)
