"""Search skills built from one primitive: a vertical peck with the closed fingertips.

A peck descends at a controlled speed until the torque-residual force says something pushed back, and reports the
TCP height where it stopped. Everything else (stack height, cup location, handle yaw) is inferred from those numbers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.kin import frame_from, rot_about
from sim_station.motion import PlanError
from sim_station.runtime import CTRL_STEPS, Ctx, SkillFault
from sim_station.sensing import Baseline
from sim_station.skills import HOLD_DZ, Z_UP, fmt_mm, lever_dirs, lever_grip_frame, side_frame
from sim_station.skills_l2 import top_face_z

PECK_Z_HI = 0.150
PECK_DESCENT_V = 0.16
PECK_THRESHOLD_N = 1.2


# A loose cup is pecked nearly straight down so the wrist housing stays above the fingers.
# The stack stands farther out; those same tilts miss inverse kinematics, so the height peck keeps the shallower set that reaches it.
CUP_TILTS = (-84.0, -78.0, -72.0, -66.0)
STACK_TILTS = (-70.0, -62.0, -78.0, -54.0)
PECK_TILTS = CUP_TILTS


def _frame_for_peck(ctx: Ctx, xy: np.ndarray, z: float, tilts=PECK_TILTS, depth: float = -0.018):
    kin = ctx.kin["cup"]
    seeds = [L.REAL_L1_CUP_WAYPOINTS["paper_hold"][:6], np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0]),
             np.array([0.3, 1.2, 0.9, 0.5, 0.0, 0.0]), np.array([0.2, 1.0, 0.5, 0.0, 0.0, 0.0]), ctx.q("cup")]
    cached = ctx.scratch.get("peck_tilt")
    order = ([cached] if cached is not None else []) + [t for t in tilts if t != cached]
    for tilt in order:
        T_hi = side_frame(kin, xy, z, off=-9.0, tilt=tilt, depth=depth)
        z_free = L.MAT_SURFACE_Z - L.pad_lowest_offset(T_hi[:3, :3])
        T_lo = side_frame(kin, xy, z_free - 0.003, off=-9.0, tilt=tilt, depth=depth)
        r1 = kin.ik_multi(T_hi, seeds)
        r2 = kin.ik_multi(T_lo, seeds + [r1.q]) if r1.ok else None
        if r1.ok and r2 is not None and r2.ok:
            ctx.scratch["peck_tilt"] = tilt
            return T_hi, T_lo, z_free, r1.q
    return None


def _s_peck_impl(ctx: Ctx, xy: np.ndarray, z_hi: float = PECK_Z_HI, v_down: float = PECK_DESCENT_V, z_stop: Optional[float] = None,
           stack_peck: bool = False):
    """Descend with closed fingertips above `xy`. Returns dict(contact, z, occupied, z_free)."""
    kin = ctx.kin["cup"]
    fr = _frame_for_peck(ctx, np.asarray(xy, dtype=float), z_hi, tilts=STACK_TILTS if stack_peck else CUP_TILTS)
    if fr is None:
        return dict(contact=False, z=float("nan"), occupied=False, z_free=float("nan"), reachable=False)
    T_hi, T_lo, z_free, q_hi = fr
    T_now = ctx.tcp("cup")
    # always transit above the cups: a sideways move at peck height drags link 4 and the housing through the cup
    T_up = T_hi.copy()
    T_up[2, 3] = max(float(T_hi[2, 3]), float(T_now[2, 3]), 0.28)
    r_up = kin.ik_multi(T_up, [q_hi, ctx.q("cup")])
    if r_up.ok and float(np.linalg.norm(T_now[:3, 3] - T_hi[:3, 3])) > 0.04:
        travel = M.via_move([ctx.q("cup"), r_up.q, q_hi], speed_scale=ctx.speed, min_T=0.7)
    else:
        travel = M.cart_move(kin, [T_now, T_hi], ctx.q("cup"), v_cart=0.28, speed_scale=ctx.speed, min_T=0.3, profile="trapezoid", max_joint_step=0.25)
    ctx.check_traj("cup", travel, grip_width=ctx.width("cup"))
    yield from ctx.play("cup", travel)
    z_bottom = (z_free - 0.0025) if z_stop is None else z_stop
    T_end = T_hi.copy()
    T_end[2, 3] = z_bottom
    down = M.cart_move(kin, [T_hi, T_end], ctx.q("cup"), v_cart=v_down, speed_scale=ctx.speed, min_T=0.4,
                       profile="trapezoid", max_joint_step=0.22)
    ctx.monitor("cup", True)
    state = {"n": 0, "t": 0}

    def on_tick(k: int) -> None:
        state["t"] = k

    def guard() -> bool:
        if state["t"] * down.dt < 0.2:                     # let the profile ramp up and the force history fill
            return False
        # push on the fingertips = upward force on the hand, seen as a step within the last ~50 ms
        if float(ctx.mon["cup"].delta_force(25)[2]) > PECK_THRESHOLD_N:
            state["n"] += 1
        else:
            state["n"] = 0
        return state["n"] >= 4

    done = yield from ctx.play("cup", down, guard=guard, on_tick=on_tick)
    z_c = float(ctx.tcp("cup")[2, 3])
    ctx.monitor("cup", False)
    contact = not done
    occupied = bool(contact and z_c > z_free + 0.007)
    k_stop = int(min(state["t"], len(down.q) - 1))
    ctx.scratch.setdefault("peck_log", []).append(dict(
        xy=np.asarray(xy, float).copy(), occupied=occupied, z=z_c, grip=float(ctx.width("cup")),
        q_stop=ctx.q("cup").copy(), q_pre=down.q[max(k_stop - 60, 0)].copy(), q_post=down.q[min(k_stop + 27, len(down.q) - 1)].copy(),
        q_bottom=down.q[-1].copy()))
    # lift the blade clear again so the next move never starts with the hand pressed on something
    if contact:
        T_c = ctx.tcp("cup")
        T_u = T_c.copy()
        T_u[2, 3] = z_hi
        try:
            up = M.cart_move(kin, [T_c, T_u], ctx.q("cup"), v_cart=0.12, speed_scale=ctx.speed, min_T=0.4, max_joint_step=0.25)
            yield from ctx.play("cup", up)
        except PlanError:
            pass
    return dict(contact=contact, z=z_c, occupied=occupied, z_free=z_free, reachable=True)


def s_peck(ctx: Ctx, *args, **kwargs):
    """A peck whose approach the planner refuses (path would hit the other arm or the bench) is reported unreachable, not fatal."""
    try:
        res = yield from _s_peck_impl(ctx, *args, **kwargs)
    except PlanError:
        ctx.q_cmd["cup"] = ctx.q("cup")
        return dict(contact=False, z=float("nan"), occupied=False, z_free=float("nan"), reachable=False)
    return res


def s_close_fingertips(ctx: Ctx):
    yield from ctx.grip_open("cup", 0.0, cap=L.FORCE_CAP_PROBE_N, tol=0.0015, timeout=3.5)
    return {"msg": "fingertips closed into a blade for pecking"}


# ---------------------------------------------------------------------------
# Level 3: how tall is the stack?
# ---------------------------------------------------------------------------

def s_find_height(ctx: Ctx):
    xy = ctx.bel.stack_xy
    z_hi = top_face_z(L.NEST_MAX) + 0.03 + 0.0095
    res = yield from s_peck(ctx, xy, z_hi=z_hi, v_down=0.06, z_stop=0.03, stack_peck=True)
    if not res["reachable"]:
        raise SkillFault("the stack top cannot be reached with the fingertips pointing forward")
    if not res["contact"]:
        raise SkillFault("the peck reached the mat without touching anything: no stack under the believed position")
    top = res["z"] + L.pad_lowest_offset(ctx.tcp("cup")[:3, :3])
    n = int(round((top - (L.MAT_SURFACE_Z + 0.0004 + L.CUP_HEIGHT)) / L.NEST_STEP)) + 1
    n = int(np.clip(n, 1, L.NEST_MAX))
    ctx.bel.stack_n = n
    ctx.bel.stack_top_z = float(top)
    # lift the blade off the stack before anything else touches the fingers
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    lifted = False
    for dz in (0.05, 0.03, 0.015):
        T1 = T0.copy()
        T1[2, 3] = float(T0[2, 3]) + dz
        try:
            up = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.4, max_joint_step=0.3)
            yield from ctx.play("cup", up)
            lifted = True
            break
        except PlanError:
            continue
    return {"msg": f"{n} cups, top at {top * 100:.1f} cm (stopped at TCP {res['z'] * 100:.1f} cm)"}


# ---------------------------------------------------------------------------
# Levels 2, 3 and 5: find an upright cup by pecking
# ---------------------------------------------------------------------------

def s_find_cup(ctx: Ctx, prior: np.ndarray, spacing: float = 0.03, iters: int = 3, ring: bool = True,
               max_pecks: int = 40):
    """Locate an upright cup near `prior` from peck outcomes only. The cup's mouth and rim stop the blade high, bare mat stops
    it low. Occupied points bracket the cup; bisection on the four edges fits the centre."""
    prior = np.asarray(prior, dtype=float)
    results: List[Tuple[np.ndarray, bool]] = []
    cache = {}

    def probe(p: np.ndarray):
        key = (round(float(p[0]) * 1000), round(float(p[1]) * 1000))
        if key in cache:
            return cache[key]
        if len(results) >= max_pecks:
            raise SkillFault(f"gave up after {max_pecks} pecks without pinning down the cup")
        r = yield from s_peck(ctx, p, z_stop=0.08)
        if not r["reachable"]:
            cache[key] = None
            return None
        results.append((p.copy(), r["occupied"]))
        cache[key] = r["occupied"]
        ctx.event(f"peck ({p[0]:.3f}, {p[1]:.3f}) -> {'cup' if r['occupied'] else 'mat'} (stopped at {r['z'] * 100:.1f} cm)")
        return r["occupied"]

    pattern = [prior] + [prior + d for d in (np.array([spacing, 0.0]), np.array([-spacing, 0.0]), np.array([0.0, spacing]), np.array([0.0, -spacing]))]
    for p in pattern:
        yield from probe(p)
    if not any(o for _, o in results) and ring:
        for ang in np.radians(np.arange(0, 360, 45)):
            yield from probe(prior + 2 * spacing * np.array([np.cos(ang), np.sin(ang)]))
    occ = [p for p, o in results if o]
    if not occ:
        raise SkillFault(f"no cup found within {2 * spacing * 100:.0f} cm of ({prior[0]:.3f}, {prior[1]:.3f})")
    c = np.mean(occ, axis=0)
    # refine: bisect the four edges through the current centre
    for axis in (0, 1):
        edges = []
        for sign in (1.0, -1.0):
            lo, hi = 0.0, 0.07
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                p = c.copy()
                p[axis] += sign * mid
                o = yield from probe(p)
                if o is None:
                    break
                if o:
                    lo = mid
                else:
                    hi = mid
            edges.append(0.5 * (lo + hi))
        c = c.copy()
        c[axis] += 0.5 * (edges[0] - edges[1])
    ctx.bel.cup_xy = c
    return {"msg": f"cup found at ({c[0]:.3f}, {c[1]:.3f}) from {len(results)} pecks"}


def _cup_violations(ctx: Ctx, c: np.ndarray, stop_at: int = 99) -> int:
    """How many peck outcomes contradict a cup standing at c: a peck that stopped high must be touching the cup at the stop
    and clear of it a few millimetres earlier; a peck that reached the mat must never have touched it."""
    P = ctx.plant
    v = 0
    for pk in ctx.scratch.get("peck_log", []):
        if pk["occupied"]:
            if not P.cup_hits_arm("cup", pk["q_post"], pk["grip"], c):
                v += 1
            if P.cup_hits_arm("cup", pk["q_pre"], pk["grip"], c):
                v += 1
        else:
            if P.cup_hits_arm("cup", pk["q_stop"], pk["grip"], c) or P.cup_hits_arm("cup", pk["q_bottom"], pk["grip"], c):
                v += 1
        if v >= stop_at:
            break
    return v


def fit_cup_position(ctx: Ctx, near: np.ndarray, half: float = 0.11) -> Tuple[np.ndarray, int, float]:
    """Centre of the set of cup positions that explain every peck (set-membership fit with the known arm and cup shapes)."""
    best = None
    for stage, (step, span, centre) in enumerate([(0.012, half, near), (0.003, 0.016, None)]):
        if centre is None:
            centre = best[0]
        grid = [centre + np.array([dx, dy]) for dx in np.arange(-span, span + 1e-9, step) for dy in np.arange(-span, span + 1e-9, step)]
        scored = []
        vmin = 99
        for c in grid:
            v = _cup_violations(ctx, c, stop_at=vmin + 1)
            vmin = min(vmin, v)
            scored.append((v, c))
        feas = np.array([c for v, c in scored if v == vmin])
        best = (feas.mean(axis=0), vmin, float(np.max(np.linalg.norm(feas - feas.mean(axis=0), axis=1))) if len(feas) else 0.0)
    return best


def _predict_hit(p: np.ndarray, c: np.ndarray, radius: float) -> bool:
    """A peck at p touches a cup centred at c when the fingertip disk overlaps the rim. Used only to choose the next peck."""
    return float(np.linalg.norm(np.asarray(p) - np.asarray(c))) < radius


RIM_INNER = 0.028


def _on_rim(p: np.ndarray, c: np.ndarray, outer: float) -> bool:
    """True when a vertical peck at p would meet the rim of a cup at c, not fall through the open mouth."""
    d = float(np.linalg.norm(np.asarray(p) - np.asarray(c)))
    return RIM_INNER < d < outer


def s_fit_cup(ctx: Ctx, near: np.ndarray, max_pecks: int = 8):
    """Fine search. The next peck is the one that splits the set of centres still consistent with the touches so far
    (the information-gain rule in the tactile-localisation papers), and it stops once that set is under 5 mm."""
    near = np.asarray(near, dtype=float)
    ctx.scratch["peck_log"] = []
    xs = np.arange(-0.07, 0.071, 0.008)
    ys = np.arange(-0.07, 0.071, 0.008)
    cand = [near + np.array([dx, dy]) for dx in xs for dy in ys]
    probed: List[np.ndarray] = []
    done = 0
    while done < max_pecks and len(cand) > 1:
        if done >= 4 and any(pk["occupied"] for pk in ctx.scratch["peck_log"]) and any(not pk["occupied"] for pk in ctx.scratch["peck_log"]):
            _c, v_try, spread_try = fit_cup_position(ctx, near)
            if v_try == 0 and spread_try < 0.005:
                break
        centre = np.mean(cand, axis=0)
        ring = [centre]
        for rad in (0.032, 0.050, 0.070):
            for ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                ring.append(centre + rad * np.array([np.cos(ang), np.sin(ang)]))
        best, best_score = None, 1e9
        n = float(len(cand))
        for p in ring:
            if any(np.linalg.norm(p - q) < 0.012 for q in probed):
                continue
            if any(np.linalg.norm(p - c0) < r - 0.02 for c0, r in L.L5_KEEPOUT):
                continue
            occ = sum(1 for c in cand if _predict_hit(p, c, 0.050))
            score = abs(occ / n - 0.5)
            if score < best_score:
                best_score, best = score, p
        if best is None:
            break
        probed.append(np.asarray(best, float))
        r = yield from s_peck(ctx, best, z_stop=0.08)
        if not r["reachable"]:
            continue
        done += 1
        ctx.event(f"peck ({best[0]:.3f}, {best[1]:.3f}) -> {'something tall' if r['occupied'] else 'mat'} (stopped at {r['z'] * 100:.1f} cm)")
        if r["occupied"]:
            cand = [c for c in cand if _predict_hit(best, c, 0.064)]
        else:
            # a miss at the 8 cm stop also happens when the blade drops into the open mouth, so that miss
            # must not delete cup centres the peck fell inside
            cand = [c for c in cand if not _on_rim(best, c, 0.040)]
        if not cand:
            cand = [np.asarray(best, float)]
            break
    if not any(pk["occupied"] for pk in ctx.scratch["peck_log"]):
        raise SkillFault("no peck around the coarse hit stopped high: nothing to fit")
    c, viol, spread = fit_cup_position(ctx, near)
    ctx.event(f"fit: cup at ({c[0]:.3f}, {c[1]:.3f}), {viol} contradicting pecks, feasible set radius {spread * 1000:.0f} mm")
    ctx.bel.cup_xy = c
    ctx.scratch["fit_xy"] = c.copy()
    return {"msg": f"cup fitted at ({c[0]:.3f}, {c[1]:.3f}) from {done} pecks (set radius {spread * 1000:.0f} mm)"}


def search_grid(ctx: Ctx) -> List[np.ndarray]:
    """Coarse grid over the free mat on the cup arm's side, skipping the places the calibration says are occupied."""
    pts: List[np.ndarray] = []
    xs = np.arange(L.L5_REGION_X[0] + 0.01, L.L5_REGION_X[1] + 1e-6, 0.060)       # 6 cm can miss a 76 mm rim; a miss is followed by two offsets in s_scan_mat
    ys = np.arange(L.L5_REGION_Y[1] - 0.01, L.L5_REGION_Y[0] - 1e-6, -0.060)
    for i, x in enumerate(xs):
        col = ys if i % 2 == 0 else ys[::-1]
        for y in col:
            p = np.array([x, y])
            if any(np.linalg.norm(p - c) < r for c, r in L.L5_KEEPOUT):
                continue
            pts.append(p)
    # spiral out from the middle of the mat so a cup near the middle is found in a handful of pecks, without dropping the corners
    mid = np.array([0.5 * (L.L5_REGION_X[0] + L.L5_REGION_X[1]), 0.5 * (L.L5_REGION_Y[0] + L.L5_REGION_Y[1])])
    pts.sort(key=lambda p: (int(np.linalg.norm(p - mid) / 0.065), np.arctan2(p[1] - mid[1], p[0] - mid[0])))
    return pts


def _scan_ok(p: np.ndarray) -> bool:
    x0, x1 = min(L.L5_REGION_X), max(L.L5_REGION_X)
    y0, y1 = min(L.L5_REGION_Y), max(L.L5_REGION_Y)
    if not (x0 <= float(p[0]) <= x1 and y0 <= float(p[1]) <= y1):
        return False
    return not any(np.linalg.norm(p - c) < r for c, r in L.L5_KEEPOUT)


def s_scan_mat(ctx: Ctx):
    """Level 5: peck a coarse grid until something stands up from the mat, then hand over to the fine search.

    The search region is L5_REGION, not the whole mat. A 6 cm cell can put its only sample in the open mouth,
    which stops at 8 cm with no contact. Each such miss is followed by two pecks 32 mm away, on the rim if the
    first peck was near the cup's centre."""
    pts = search_grid(ctx)
    done = 0

    def one(p: np.ndarray):
        nonlocal done
        r = yield from s_peck(ctx, p, z_stop=0.08)
        done += 1
        if r["reachable"] and r["occupied"]:
            ctx.bel.found["coarse_hit"] = np.asarray(p, float).copy()
            ctx.event(f"coarse hit at ({p[0]:.3f}, {p[1]:.3f}): blade stopped at {r['z'] * 100:.1f} cm after {done} pecks")
            return True
        return False

    for p in pts:
        if (yield from one(p)):
            return {"msg": f"something stands at ({p[0]:.3f}, {p[1]:.3f}) after {done} pecks"}
        for d in ((0.032, 0.0), (0.0, 0.032)):
            q = np.asarray(p, float) + np.array(d)
            if not _scan_ok(q):
                continue
            if (yield from one(q)):
                return {"msg": f"something stands at ({q[0]:.3f}, {q[1]:.3f}) after {done} pecks"}
    raise SkillFault(f"scanned {done} grid points and found nothing standing on the mat: the cup is outside the searched region or the pecks missed its rim")


# ---------------------------------------------------------------------------
# Level 4: where is the handle? Sweep a fingertip around the valve, feel it, push it to a known yaw, feel it again.
# ---------------------------------------------------------------------------

SWEEP_RADIUS = 0.034
SWEEP_Z = L.LEVER_PIVOT_Z + 0.0015 + 0.5 * L.LEVER_HANDLE_THICK + 0.002
PARK_YAW = L.LEVER_YAW_DEFAULT
PARK_TOLERANCE = np.radians(8.0)
PARK_OVERSHOOT = np.radians(4.0)            # a pushed handle stalls a little short: aim a little beyond

_T = L.LEVER_HANDLE_THICK
_ZC = 0.0015 + 0.5 * _T
# collision boxes of the handle in its own frame (stem and paddle, from the model of the dispenser)
HANDLE_BOXES = [
    (np.array([0.008, 0.0, _ZC]), np.array([0.010, 0.5 * 0.011, 0.5 * _T])),
    (np.array([0.026, 0.0, _ZC]), np.array([0.5 * (L.LEVER_HANDLE_LEN - 0.008), 0.5 * L.LEVER_HANDLE_WIDTH, 0.5 * _T])),
]


def obb_overlap(c1: np.ndarray, R1: np.ndarray, h1: np.ndarray, c2: np.ndarray, R2: np.ndarray, h2: np.ndarray) -> bool:
    """Separating-axis test of two oriented boxes (centre, rotation with the box axes as columns, half sizes)."""
    d = c2 - c1
    C = R1.T @ R2
    A = np.abs(C) + 1e-9
    t = R1.T @ d
    if np.any(np.abs(t) > h1 + A @ h2):
        return False
    if np.any(np.abs(R2.T @ d) > h2 + A.T @ h1):
        return False
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = h1[i1] * A[i2, j] + h1[i2] * A[i1, j]
            rb = h2[j1] * A[i, j2] + h2[j2] * A[i, j1]
            if abs(t[i2] * C[i1, j] - t[i1] * C[i2, j]) > ra + rb:
                return False
    return True


def hand_boxes_world(T: np.ndarray, width: float):
    """The two fingertip pads (tip and lead-in funnel) in the world for TCP pose T and jaw opening `width`."""
    out = []
    for side, _name, c, R, h in L.pad_boxes_site():
        sgn = 1.0 if side == "left" else -1.0
        cs = c + np.array([0.0, sgn * 0.5 * width, 0.0])
        out.append((T[:3, 3] + T[:3, :3] @ cs, T[:3, :3] @ R, np.asarray(h, dtype=float)))
    return out


def handle_boxes_world(pivot: np.ndarray, psi: float):
    c, s = np.cos(psi), np.sin(psi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return [(pivot + Rz @ cl, Rz, h) for cl, h in HANDLE_BOXES]


def _first_hit(hit, start: float, step: float, n: int):
    """March from `start` in increments of `step`; return the parameter where `hit` first becomes true (bisected)."""
    prev = None
    for i in range(n):
        x = start + i * step
        if hit(x):
            if prev is None:
                return x
            lo, hi = x, prev
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if hit(mid):
                    lo = mid
                else:
                    hi = mid
            return lo
        prev = x
    return None


def touch_yaw(ctx: Ctx, T: np.ndarray, width: float, side: float) -> Optional[float]:
    """Yaw the handle must have for the fingertip at pose T to be just touching it, the handle lying ahead of the
    finger in direction `side` (+1 counter-clockwise). Uses the known hand and handle shapes, not a fixed offset."""
    P = ctx.bel.lever_pivot
    r = T[:3, 3] - P
    th = float(np.arctan2(r[1], r[0]))
    hand = hand_boxes_world(T, width)

    def hit(psi: float) -> bool:
        hb = handle_boxes_world(P, psi)
        return any(obb_overlap(*a, *b) for a in hand for b in hb)

    return _first_hit(hit, th + side * np.radians(115.0), -side * np.radians(1.0), 140)


def touch_theta(ctx: Ctx, psi: float, side: float, R_tool: np.ndarray, width: float) -> Optional[float]:
    """Finger angle about the valve at which the fingertip touches a handle lying at yaw psi (approaching from `side`)."""
    P = ctx.bel.lever_pivot
    hb = handle_boxes_world(P, psi)

    def hit(th: float) -> bool:
        hand = hand_boxes_world(_sweep_pose(ctx, th, R_tool), width)
        return any(obb_overlap(*a, *b) for a in hand for b in hb)

    return _first_hit(hit, psi - side * np.radians(115.0), side * np.radians(1.0), 140)


def _sweep_pose(ctx: Ctx, theta: float, R_tool: np.ndarray, radius: float = SWEEP_RADIUS) -> np.ndarray:
    P = ctx.bel.lever_pivot
    T = np.eye(4)
    T[:3, :3] = R_tool
    T[:3, 3] = np.array([P[0] + radius * np.cos(theta), P[1] + radius * np.sin(theta), SWEEP_Z])
    return T


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def s_find_lever(ctx: Ctx):
    kin = ctx.kin["tap"]
    P = ctx.bel.lever_pivot
    R_tool = lever_grip_frame(P, PARK_YAW)[:3, :3]
    q0 = ctx.q("tap")
    seeds = [np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.5, 1.2, 0.8, 0.6, 0.0, 0.0]), np.array([0.3, 2.0, 1.6, 1.4, 0.2, 0.0]), q0]
    yield from ctx.grip_open("tap", 0.0, cap=L.FORCE_CAP_PROBE_N, tol=0.0015, timeout=3.5)
    th_lo = np.radians(-170.0)
    th_hi = np.radians(190.0)                     # a full circle: the swivel has no stop

    def arc_poses(a: float, b: float, step: float = 0.12):
        n = max(int(abs(b - a) / step), 2)
        return [_sweep_pose(ctx, th, R_tool) for th in np.linspace(a, b, n + 1)]

    def finger_angle() -> float:
        r = ctx.mon["tap"].tcp[:3, 3] - P
        return float(np.arctan2(r[1], r[0]))

    def go_above(th: float):
        """Lift off the plane, hop through the air above the handle and come down at finger angle th."""
        T_start = _sweep_pose(ctx, th, R_tool)
        T_above = T_start.copy()
        T_above[2, 3] += 0.07
        rs = kin.ik_multi(T_start, seeds + [ctx.q("tap")])
        ra = kin.ik_multi(T_above, seeds + [rs.q, ctx.q("tap")])
        if not (rs.ok and ra.ok):
            raise SkillFault(f"the sweep start is unreachable for the tap arm ({rs.reason or ra.reason})")
        here = ctx.tcp("tap")
        T_up = here.copy()
        T_up[2, 3] = max(here[2, 3], SWEEP_Z) + 0.06
        if np.linalg.norm(here[:3, 3] - T_start[:3, 3]) > 0.03 and here[2, 3] < SWEEP_Z + 0.03 and np.linalg.norm(here[:3, 3] - P) < 0.1:
            lift = M.cart_move(kin, [here, T_up], ctx.q("tap"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.6)
            yield from ctx.play("tap", lift)
        traj = M.via_move([ctx.q("tap"), ra.q], speed_scale=ctx.speed, min_T=1.2)
        ctx.check_traj("tap", traj, grip_width=0.0)
        yield from ctx.play("tap", traj)
        # come down onto the sweep plane slowly; if something under the fingertip pushes back it is the handle, so land elsewhere
        ctx.monitor("tap", True)
        for attempt in range(8):
            th_try = th + attempt * np.radians(26.0)
            if attempt:
                T_s = _sweep_pose(ctx, th_try, R_tool)
                T_s[2, 3] += 0.07
                rr = kin.ik_multi(T_s, seeds + [ctx.q("tap")])
                if not rr.ok:
                    continue
                yield from ctx.play("tap", M.joint_move(ctx.q("tap"), rr.q, speed_scale=ctx.speed, min_T=0.6))
            T_here = _sweep_pose(ctx, th_try, R_tool)
            T_top = T_here.copy()
            T_top[2, 3] += 0.07
            rd = kin.ik_multi(T_here, seeds + [ctx.q("tap")])
            if not rd.ok:
                continue
            down = M.cart_move(kin, [ctx.tcp("tap"), T_here], ctx.q("tap"), v_cart=0.03, speed_scale=ctx.speed, min_T=1.0, profile="trapezoid")
            hit = {"n": 0}

            def g() -> bool:
                hit["n"] = hit["n"] + 1 if float(ctx.mon["tap"].delta_force(25)[2]) > 0.9 else 0
                return hit["n"] >= 4

            done = yield from ctx.play("tap", down, guard=g)
            if done:
                break
            ctx.event(f"the fingertip landed on the handle at {np.degrees(th_try):.0f} deg; lifting and landing {26} degrees on")
            yield from ctx.play("tap", M.cart_move(kin, [ctx.tcp("tap"), T_top], ctx.q("tap"), v_cart=0.06, speed_scale=ctx.speed, min_T=0.5))
        ctx.monitor("tap", False)
        return th_try

    def sweep(a: float, b: float, direction: float, min_T: float = 2.0, win: Tuple[float, float] = (0.35, 1.0)):
        """Move the fingertip along the arc a -> b until the torque residual says something pushed back.
        Returns (finger angle at contact or None, completed)."""
        poses = [ctx.tcp("tap")] + arc_poses(a, b)
        tr = M.cart_move(kin, poses, ctx.q("tap"), v_cart=L.SWEEP_SPEED_M_S, speed_scale=ctx.speed, min_T=min_T, step_m=0.003,
                         profile="trapezoid", max_joint_step=0.2)
        ctx.check_traj("tap", tr, grip_width=0.0)
        base = Baseline()
        st = {"n": 0, "theta": None}

        def tangent_at() -> Tuple[float, np.ndarray]:
            th = finger_angle()
            return th, np.array([-np.sin(th), np.cos(th), 0.0])

        def on_tick(k: int) -> None:
            if win[0] < k * tr.dt < win[1]:
                base.add(ctx.mon["tap"].force)

        def guard() -> bool:
            if len(base.samples) < 40:
                return False
            mean, std = base.stats()
            th, tang = tangent_at()
            push = float(-(ctx.mon["tap"].force - mean) @ tang) * direction
            thr = max(0.9, 6 * float(np.max(std)))
            ctx.guard_state.update(value=push, threshold=thr, label="finger push")
            if push > thr:
                st["n"] += 1
            else:
                st["n"] = 0
            if st["n"] >= 6:
                st["theta"] = th
                return True
            return False

        done = yield from ctx.play("tap", tr, guard=guard, on_tick=on_tick)
        return st["theta"], done

    def probe(side: float, th_from: float):
        """Back the finger off from the handle, then come at it again slowly; return the yaw the touch implies."""
        th_back = th_from - side * np.radians(34.0)
        back = M.cart_move(kin, [ctx.tcp("tap"), _sweep_pose(ctx, th_back, R_tool)], ctx.q("tap"), v_cart=0.04, speed_scale=ctx.speed, min_T=0.7)
        yield from ctx.play("tap", back)
        ctx.monitor("tap", True)
        th_t, _ = yield from sweep(th_back, th_from + side * np.radians(16.0), side, min_T=2.6, win=(0.3, 0.75))
        if th_t is None:
            return None
        return touch_yaw(ctx, ctx.mon["tap"].tcp, ctx.width("tap"), side)

    # --- 1. go round counter-clockwise from the start until the fingertip touches the handle
    side = +1.0

    def locate():
        nonlocal side
        t0 = yield from go_above(th_lo)
        ctx.monitor("tap", True)
        th_c, _ = yield from sweep(t0, t0 + np.radians(355.0), +1.0)
        if th_c is None:
            raise SkillFault("swept the whole way round the valve and never touched the handle")
        y = touch_yaw(ctx, ctx.mon["tap"].tcp, ctx.width("tap"), +1.0)
        if y is None:
            raise SkillFault("felt something at the valve but it does not match the shape of the handle")
        side = +1.0
        ctx.event(f"handle felt with the fingertip at {np.degrees(th_c):.0f} deg: it lies at about {np.degrees(_wrap(y)):.0f} deg")
        return _wrap(y)

    yaw = yield from locate()
    yaw_first = yaw

    # --- 2. put it where the hand can take it comfortably, and measure again by touch each time
    for attempt in range(5):
        err = _wrap(yaw - PARK_YAW)
        if abs(err) <= PARK_TOLERANCE:
            break
        want = 1.0 if err < 0 else -1.0
        if want != side:
            # we are standing on the wrong side of the handle: hop over it and come back the other way
            far = yaw - want * np.radians(55.0)             # start 55 degrees short of the handle, on the side we want to push from
            far = yield from go_above(far)
            ctx.monitor("tap", True)
            th_c, _ = yield from sweep(far, far + want * np.radians(130.0), want)
            if th_c is None:
                raise SkillFault("lost the handle while coming round to the other side")
            y2 = touch_yaw(ctx, ctx.mon["tap"].tcp, ctx.width("tap"), want)
            if y2 is not None:
                yaw = _wrap(y2)
            side = want
            ctx.event(f"touched the handle from the other side: it lies at about {np.degrees(yaw):.0f} deg")
            if abs(_wrap(yaw - PARK_YAW)) <= PARK_TOLERANCE:
                break
        th_goal = touch_theta(ctx, PARK_YAW + want * PARK_OVERSHOOT, want, R_tool, ctx.width("tap"))
        if th_goal is None:
            raise SkillFault("cannot work out where to hold the fingertip to park the handle")
        th_now = finger_angle()
        th_goal = th_now + want * ((want * (th_goal - th_now)) % (2.0 * np.pi))       # travel in the pushing direction, never the long way round
        poses = [ctx.tcp("tap")] + arc_poses(th_now, th_goal, 0.08)
        tr = M.cart_move(kin, poses, ctx.q("tap"), v_cart=L.SWEEP_SPEED_M_S, speed_scale=ctx.speed, min_T=1.0, step_m=0.003, max_joint_step=0.2)
        ctx.check_traj("tap", tr, grip_width=0.0)
        yield from ctx.play("tap", tr)
        yield from ctx.settle("tap", 0.0015, 3.5)
        y_v = yield from probe(want, finger_angle())
        if y_v is None:
            ctx.event("could not feel the handle again after pushing it: it slipped past the fingertip, so start the search again")
            yaw = yield from locate()
            continue
        yaw = _wrap(y_v)
        ctx.event(f"touched it again: the handle now sits at {np.degrees(yaw):.1f} deg")
    ctx.monitor("tap", False)
    ctx.guard_state.update(value=0.0, threshold=0.0, label="")
    # lift the finger clear
    T0 = ctx.tcp("tap")
    T1 = T0.copy()
    T1[2, 3] += 0.06
    up = M.cart_move(kin, [T0, T1], ctx.q("tap"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.6)
    yield from ctx.play("tap", up)
    ctx.bel.lever_yaw = float(yaw)
    ctx.bel.found["lever_yaw_before"] = float(yaw_first)
    return {"msg": f"handle was at about {np.degrees(yaw_first):.0f} deg; now at {np.degrees(yaw):.0f} deg, checked by touch"}
