"""Search skills. Level 3 pecks the stack with a fingertip blade. Level 5 does not.

Level 5 opens the jaws and lays the black pads flat on the mat, the way a hand slaps with the whole
palm rather than one fingertip. The left arm stomps the +y half, the right arm the -y half. A hit is
the near rim, not the cup: the arm walks the patch until empty mat brackets it, and the centre of that
patch is the cup.

A guarded descent stops when the torque residual says something pushed back, and reports the TCP height.
Stack height, cup location and handle yaw are inferred from those numbers. The skills never read a pose.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.beliefgeom import feasible_region
from sim_station.kin import frame_from, rot_about
from sim_station.motion import PlanError
from sim_station.runtime import CTRL_STEPS, Ctx, SkillFault
from sim_station.sensing import Baseline
from sim_station.skills import (HOLD_DZ, LEVER_GRIP_Z, LEVER_OPEN_THETA, LEVER_SEEDS, Z_UP, fmt_mm, lever_dirs,
                                lever_frame, lever_grip_frame, s_goto_rest, side_frame)
from sim_station.skills_l2 import top_face_z

PECK_Z_HI = 0.150
PECK_DESCENT_V = 0.16
PECK_THRESHOLD_N = 1.2


# A loose cup is pecked nearly straight down so the wrist housing stays above the fingers.
# The stack stands farther out; those same tilts miss inverse kinematics, so the height peck keeps the shallower set that reaches it.
CUP_TILTS = (-84.0, -78.0, -72.0, -66.0)
STACK_TILTS = (-70.0, -62.0, -78.0, -54.0)
# Level 5 spreads the jaws. The pads still come down steeply, or a shallow hand misses the cup
# it is aiming at. The width, not a fingertip point, is what covers the rim.
STAMP_TILTS = (-78.0, -72.0, -84.0, -66.0)
STAMP_WIDTH = 0.062
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
            if ctx.scratch.get("probe") == "stamp":
                # Put the midpoint of the two fingertip pads over the query, not the wrist.
                mid = np.array([0.0, 0.0, -L.PAD_TIP_HALF[2]])
                for T in (T_hi, T_lo):
                    T[:2, 3] -= (T[:3, :3] @ mid)[:2]
                z_free = L.MAT_SURFACE_Z - L.pad_lowest_offset(T_hi[:3, :3])
                T_lo[2, 3] = z_free - 0.003
                r1 = kin.ik_multi(T_hi, seeds + [r1.q])
                r2 = kin.ik_multi(T_lo, seeds + [r1.q]) if r1.ok else None
                if not (r1.ok and r2 is not None and r2.ok):
                    continue
            ctx.scratch["peck_tilt"] = tilt
            return T_hi, T_lo, z_free, r1.q
    return None


def _s_peck_impl(ctx: Ctx, xy: np.ndarray, z_hi: float = PECK_Z_HI, v_down: float = PECK_DESCENT_V, z_stop: Optional[float] = None,
           stack_peck: bool = False):
    """Descend with closed fingertips above `xy`. Returns dict(contact, z, occupied, z_free)."""
    kin = ctx.kin["cup"]
    if ctx.scratch.get("probe") == "stamp":
        v_down = min(float(v_down), 0.07)
    fr = _frame_for_peck(ctx, np.asarray(xy, dtype=float), z_hi,
                         tilts=STACK_TILTS if stack_peck else (STAMP_TILTS if ctx.scratch.get("probe") == "stamp" else CUP_TILTS))
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
    ctx.scratch["probe"] = "blade"
    return {"msg": "fingertips closed into a blade for pecking"}


def s_open_stamp(ctx: Ctx):
    """Spread the jaws and come down with the hand flatter, so the two pads cover a patch of the mat."""
    w = yield from ctx.grip_open("cup", STAMP_WIDTH, cap=L.FORCE_CAP_PROBE_N, tol=0.002, timeout=3.5)
    ctx.scratch["probe"] = "stamp"
    return {"msg": f"jaws at {fmt_mm(w)}, hand low: a stamp across the rim, not a fingertip point"}


# ---------------------------------------------------------------------------
# Level 5: flat-hand stomp, one half of the mat per arm
# ---------------------------------------------------------------------------

STOMP_WIDTH = 0.064
# tilt 0 lays the black pad bottoms parallel to the mat (TCP x straight down). The wrist housing is
# lower than the pads, but it trails a hand's length behind them, over ground already stomped, and the
# descent stops at the rim so the housing never reaches the mat or a cup the pads have not already met.
STOMP_TILTS = (0.0, -8.0, -15.0, -24.0)
# Pitched up far enough that the housing clears a rim. Used only where a cup could still be under the housing.
SAFE_TILTS = (-32.0, -38.0, -44.0)
STOMP_Z = 0.20
# Pad bottoms stop just under the rim. Deeper, and a pad that only clips the wall shoves the cup sideways.
STOMP_PAD_FLOOR = 0.104


def _half_split() -> float:
    return float(0.5 * (L.L5_REGION_Y[0] + L.L5_REGION_Y[1]))


def _stomp_seeds(ctx: Ctx, arm: str) -> list:
    if arm == "cup":
        return [ctx.q(arm), L.REAL_L1_CUP_WAYPOINTS["paper_hold"][:6],
                np.array([0.0, 1.5, 1.2, 0.2, -1.3, 0.0]), np.array([0.4, 1.4, 1.1, 0.0, -1.1, 0.3]),
                np.array([-0.3, 1.5, 1.2, 0.2, 1.2, -0.2])]
    return [ctx.q(arm), np.array([0.3, 1.5, 1.2, 0.2, -1.2, 0.0]), np.array([0.6, 1.6, 1.2, 0.3, -1.0, 0.2]),
            np.array([-0.2, 1.5, 1.1, 0.2, 1.1, -0.2]), np.array([0.8, 1.3, 0.9, 0.0, -0.8, 0.4])]


def _solve_stomp(ctx: Ctx, arm: str, xy: np.ndarray):
    """A flat pose whose descent stays on one inverse-kinematics branch.

    A straight-line move onto a different wrist branch reaches the same hand pose and then cannot
    come back down: joint 4 is already on its stop. Only a solution that can descend from its own
    joint vector is kept.
    """
    kin = ctx.kin[arm]
    seeds = _stomp_seeds(ctx, arm) + [ctx.q(arm)]
    cached = ctx.scratch.get(f"stomp_tilt_{arm}")
    order = ([cached] if cached is not None else []) + [t for t in STOMP_TILTS if t != cached] + list(SAFE_TILTS)
    xy = np.asarray(xy, float)
    for tilt in order:
        for off in (0.0, -16.0, 16.0, -32.0, 32.0, 50.0, -50.0):
            T = side_frame(kin, xy, STOMP_Z, off=off, tilt=tilt, depth=0.0)
            z_floor = STOMP_PAD_FLOOR - float(L.pad_lowest_offset(T[:3, :3]))
            T_hi = T.copy()
            T_hi[2, 3] = STOMP_Z
            T_lo = T.copy()
            T_lo[2, 3] = z_floor
            if _housing_risk(ctx, T_lo):
                continue
            rh = kin.ik_multi(T_hi, seeds)
            if not rh.ok:
                continue
            try:
                M.cart_move(kin, [T_hi, T_lo], rh.q, v_cart=0.05, speed_scale=ctx.speed,
                            min_T=0.30, profile="trapezoid", max_joint_step=0.22)
            except PlanError:
                continue
            if tilt in STOMP_TILTS:
                ctx.scratch[f"stomp_tilt_{arm}"] = tilt
            return T_hi, T_lo, rh.q, z_floor
    return None


def _arrive_flat(ctx: Ctx, arm: str, T_hi: np.ndarray, q_hi: np.ndarray) -> bool:
    """Stand at q_hi, the joint solution the descent was planned on. A cartesian arrival on another branch does not count."""
    kin = ctx.kin[arm]
    if float(np.linalg.norm(ctx.q(arm) - q_hi)) < 0.18:
        return True
    here = ctx.tcp(arm)
    T_up = here.copy()
    T_up[2, 3] = max(float(here[2, 3]), float(T_hi[2, 3]))
    try:
        travel = M.cart_move(kin, [here, T_up, T_hi], ctx.q(arm), v_cart=0.22, speed_scale=ctx.speed,
                             min_T=0.30, max_joint_step=0.35, profile="trapezoid")
        if float(np.linalg.norm(travel.q[-1] - q_hi)) < 0.45:
            ctx.check_traj(arm, travel, grip_width=ctx.width(arm))
            yield from ctx.play(arm, travel)
            return True
    except PlanError:
        pass
    # A joint hop after the cup has been touched swings the wrist housing through it. Cartesian only from then on.
    if ctx.scratch.get("stomp_hits"):
        return False
    try:
        hop = M.joint_move(ctx.q(arm), q_hi, speed_scale=ctx.speed, min_T=0.55)
        ctx.check_traj(arm, hop, grip_width=ctx.width(arm))
        yield from ctx.play(arm, hop)
        return True
    except PlanError:
        return False


def _s_stomp(ctx: Ctx, arm: str, xy: np.ndarray):
    """One flat stomp. Occupied when the pads stop on something standing up: a cup, not the mat."""
    kin = ctx.kin[arm]
    solved = _solve_stomp(ctx, arm, xy)
    if solved is None:
        return dict(occupied=False, reachable=False, z=float("nan"))
    T_hi, T_lo, q_hi, z_floor = solved
    if not (yield from _arrive_flat(ctx, arm, T_hi, q_hi)):
        return dict(occupied=False, reachable=False, z=float("nan"))
    try:
        down = M.cart_move(kin, [kin.fk(q_hi), T_lo], q_hi, v_cart=0.05, speed_scale=ctx.speed,
                           min_T=0.30, profile="trapezoid", max_joint_step=0.22)
        ctx.check_traj(arm, down, grip_width=ctx.width(arm))
    except PlanError:
        return dict(occupied=False, reachable=False, z=float("nan"))
    ctx.monitor(arm, True)
    state = {"n": 0, "t": 0}

    def on_tick(k: int, state=state) -> None:
        state["t"] = k

    def guard(state=state) -> bool:
        # Ignore the descent until the command is at the rim. Acceleration up high is not a cup,
        # and a stop recorded there is what pulled the fitted centre off the real one.
        if state["t"] * down.dt < 0.12:
            return False
        k = min(state["t"], len(down.q) - 1)
        z_cmd = float(kin.fk(down.q[k])[2, 3])
        if z_cmd > z_floor + 0.018:
            return False
        if float(ctx.mon[arm].delta_force(25)[2]) > PECK_THRESHOLD_N:
            state["n"] += 1
        else:
            state["n"] = 0
        if state["n"] >= 3:
            return True
        z_now = float(ctx.tcp(arm)[2, 3])
        return bool(z_now > z_floor + 0.008 and (z_now - z_cmd) > 0.006)

    done = yield from ctx.play(arm, down, guard=guard, on_tick=on_tick)
    z_c = float(ctx.tcp(arm)[2, 3])
    ctx.monitor(arm, False)
    # Empty mat is tracked onto the floor. A cup holds the pads about a centimetre above it.
    # A stop still up at the transit height is a wrist limit, not a cup.
    occupied = bool(z_floor + 0.006 < z_c < z_floor + 0.040)
    T_at = T_lo.copy()
    T_at[:2, 3] = ctx.tcp(arm)[:2, 3]
    _map_update(ctx, T_at, float(ctx.width(arm)), occupied)
    _cup_map(ctx)["log"][-1] += (z_c - z_floor, not done)
    if occupied:
        ctx.scratch.setdefault("stomp_hits", []).append(np.asarray(xy, float).copy())
    # Back to the same branch, high, so the next cell is not planned from a wrist pinned on its stop.
    try:
        up = M.cart_move(kin, [ctx.tcp(arm), T_hi], ctx.q(arm), v_cart=0.16, speed_scale=ctx.speed,
                         min_T=0.22, max_joint_step=0.30)
        yield from ctx.play(arm, up)
    except PlanError:
        if not ctx.scratch.get("stomp_hits"):
            try:
                hop = M.joint_move(ctx.q(arm), q_hi, speed_scale=ctx.speed, min_T=0.4)
                ctx.check_traj(arm, hop, grip_width=ctx.width(arm))
                yield from ctx.play(arm, hop)
            except PlanError:
                pass
    return dict(occupied=occupied, reachable=True, z=z_c)


def _stomp_grid(arm: str) -> List[np.ndarray]:
    """3 cm cells, walked outward from that arm's base. The wrist trails over ground already stomped."""
    step = 0.030
    xs = np.arange(L.L5_REGION_X[0] + 0.015, L.L5_REGION_X[1] - 0.005, step)
    ys = np.arange(L.L5_REGION_Y[0] + 0.015, L.L5_REGION_Y[1] - 0.005, step)
    mid = _half_split()
    pts: List[np.ndarray] = []
    for x in xs:
        for y in ys:
            if arm == "cup" and float(y) < mid:
                continue
            if arm == "tap" and float(y) >= mid:
                continue
            p = np.array([float(x), float(y)])
            if _scan_ok(p):
                pts.append(p)
    base = np.asarray(L.MOUNTS[arm][0][:2], float)
    pts.sort(key=lambda p: float(np.linalg.norm(p - base)))
    return pts


def _fold_clear(ctx: Ctx, arm: str):
    try:
        yield from s_goto_rest(ctx, arm)
    except (PlanError, SkillFault):
        try:
            traj = M.joint_move(ctx.q(arm), L.REST_Q, speed_scale=ctx.speed, min_T=1.0)
            ctx.check_traj(arm, traj, grip_width=ctx.width(arm))
            yield from ctx.play(arm, traj)
            ctx.q_cmd[arm] = L.REST_Q.copy()
        except PlanError:
            ctx.event(f"could not fold the {arm} arm clear; leaving it where the last stomp finished")


def s_flat_hand(ctx: Ctx, arm: str):
    """Open the jaws so the two black pads are one flat slap, not a fingertip."""
    w = yield from ctx.grip_open(arm, STOMP_WIDTH, cap=L.FORCE_CAP_PROBE_N, tol=0.002, timeout=3.5)
    ctx.scratch["probe"] = "stomp"
    side = "left" if arm == "cup" else "right"
    return {"msg": f"{side} gripper open to {fmt_mm(w)}, black pads ready to stomp flat"}


def _in_half(arm: str, p: np.ndarray, margin: float = 0.09) -> bool:
    """A cup on the split line belongs to whichever arm touched it. The bracket may cross by `margin`."""
    mid = _half_split()
    if arm == "cup":
        return float(p[1]) >= mid - margin
    return float(p[1]) <= mid + margin


# The rim the pads can land on, at the height the descent stops: outer wall a little under the rolled bead,
# inner edge one wall in. A pad wholly inside the mouth drops into the cup, which reads as empty mat.
RIM_R_OUT = L.CUP_RIM_RADIUS + 0.0005
RIM_R_IN = L.CUP_RIM_RADIUS - L.CUP_WALL_THICK - 0.0015


# Only pad material within this height of the hand's lowest point can hold the hand far enough above its floor
# (6 mm of a 9 mm rim) to read as a cup. A pitched pad that meets the rim higher up slides off and reads as mat.
PAD_SLICE = 0.004
_U = np.linspace(-1.0, 1.0, 5)
_BOX_SAMPLES = np.stack(np.meshgrid(_U, _U, _U, indexing="ij"), axis=-1).reshape(-1, 3)


def _pad_footprints(T: np.ndarray, width: float) -> List[np.ndarray]:
    """Plan view of the part of each black pad that can land on a rim, for a hand at T with the jaws open to `width`."""
    boxes = []
    for side, _name, c, R, h in L.pad_boxes_site():
        sgn = 1.0 if side == "left" else -1.0
        cs = c + np.array([0.0, sgn * 0.5 * width, 0.0])
        local = cs + (_BOX_SAMPLES * np.asarray(h)) @ R.T
        boxes.append(local @ T[:3, :3].T + T[:3, 3])
    z_low = min(float(b[:, 2].min()) for b in boxes)
    out = []
    for b in boxes:
        sel = b[b[:, 2] <= z_low + PAD_SLICE][:, :2]
        if len(sel) >= 3:
            hull = _hull(sel)
            if len(hull) >= 3:
                out.append(hull)
    return out


def _hull(P: np.ndarray) -> np.ndarray:
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h: list = []
        for p in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h

    lo, hi = half(P), half(P[::-1])
    return np.asarray(lo[:-1] + hi[:-1])


def _poly_r_range(poly: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest and farthest distance from each centre in C (N, 2) to a convex counter-clockwise polygon (0 when inside)."""
    rel = poly[None, :, :] - C[:, None, :]
    r_max = np.max(np.linalg.norm(rel, axis=2), axis=1)
    a = rel
    b = np.roll(rel, -1, axis=1)
    e = b - a
    cross = e[..., 0] * (-a[..., 1]) - e[..., 1] * (-a[..., 0])
    inside = np.all(cross >= 0.0, axis=1)
    t = np.clip(-np.sum(a * e, axis=2) / np.maximum(np.sum(e * e, axis=2), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(a + t[..., None] * e, axis=2)
    r_min = np.where(inside, 0.0, np.min(d, axis=1))
    return r_min, r_max


def _rim_margin(polys: List[np.ndarray], C) -> np.ndarray:
    """Signed distance each centre could move before the prediction flips: > 0 means a pad lands on the rim."""
    C = np.atleast_2d(np.asarray(C, float))
    best = np.full(len(C), -0.05)
    for poly in polys:
        r_min, r_max = _poly_r_range(poly, C)
        hit = (r_min <= RIM_R_OUT) & (r_max >= RIM_R_IN)
        m = np.where(hit, np.minimum(RIM_R_OUT - r_min, r_max - RIM_R_IN),
                     np.where(r_min > RIM_R_OUT, -(r_min - RIM_R_OUT), -(RIM_R_IN - r_max)))
        best = np.maximum(best, m)
    return best


# Wrist housing collision box in the TCP frame (scene._add_collision). At tilt 0 its bottom hangs 32 mm under the TCP,
# below a cup rim, so a flat stomp is only taken where that box cannot come down on a cup the map has not ruled out.
HOUSING_C = np.array([0.0, 0.0, -0.1185])
HOUSING_HALF = np.array([0.032, 0.050, 0.037])
RIM_TOP_Z = L.MAT_SURFACE_Z + L.CUP_HEIGHT + 0.004
HOUSING_CLEAR = L.CUP_RIM_RADIUS + 0.006
MAP_STEP = 0.003
MAP_TOL = 0.004          # centres the housing must still respect
FIT_TOL = 0.002          # centres still competing for the estimate


def _housing_poly(T: np.ndarray) -> Optional[np.ndarray]:
    """Plan view of the housing for a hand at T, or None when the whole box is above the rim."""
    P = np.asarray([T[:3, :3] @ (HOUSING_C + HOUSING_HALF * np.array([sx, sy, sz])) + T[:3, 3]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if float(P[:, 2].min()) > RIM_TOP_Z:
        return None
    return _hull(P[:, :2])


def _cup_map(ctx: Ctx) -> dict:
    """Every centre the cup may have: the search region less the keep-outs. Each stomp charges the centres it contradicts."""
    m = ctx.scratch.get("cup_map")
    if m is None:
        xs = np.arange(L.L5_REGION_X[0], L.L5_REGION_X[1] + 1e-9, MAP_STEP)
        ys = np.arange(L.L5_REGION_Y[0], L.L5_REGION_Y[1] + 1e-9, MAP_STEP)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        C = np.stack([gx.ravel(), gy.ravel()], axis=1)
        ok = np.ones(len(C), bool)
        for c, r in L.L5_KEEPOUT:
            ok &= np.linalg.norm(C - np.asarray(c, float), axis=1) > r
        m = dict(C=C[ok], cost=np.zeros(int(ok.sum())))
        ctx.scratch["cup_map"] = m
    return m


def _map_update(ctx: Ctx, T: np.ndarray, width: float, occupied: bool) -> None:
    m = _cup_map(ctx)
    m.setdefault("log", []).append((T.copy(), float(width), bool(occupied)))
    mm = _rim_margin(_pad_footprints(T, width), m["C"])
    wrong = (mm > 0) != occupied
    # A light cup can skate off a pad that lands on the bead, so one stomp is never allowed to veto a centre outright.
    m["cost"] += np.where(wrong, np.minimum(np.abs(mm), 0.003), 0.0)


def _possible(ctx: Ctx, tol: float = MAP_TOL) -> np.ndarray:
    m = _cup_map(ctx)
    return m["C"][m["cost"] <= float(m["cost"].min()) + tol]


def _housing_risk(ctx: Ctx, T_lo: np.ndarray) -> bool:
    poly = _housing_poly(T_lo)
    if poly is None:
        return False
    r_min, _ = _poly_r_range(poly, _possible(ctx))
    return bool(np.any(r_min < HOUSING_CLEAR))


def _predict(ctx: Ctx, arm: str, p: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Rim margin for each candidate centre if the flat hand came down at p."""
    kin = ctx.kin[arm]
    tilt = ctx.scratch.get(f"stomp_tilt_{arm}", 0.0)
    for t in (tilt,) + tuple(SAFE_TILTS):
        T = side_frame(kin, np.asarray(p, float), 0.0, off=0.0, tilt=t)
        T[2, 3] = STOMP_PAD_FLOOR - float(L.pad_lowest_offset(T[:3, :3]))
        if not _housing_risk(ctx, T):
            break
    return _rim_margin(_pad_footprints(T, float(ctx.width(arm))), P)


def _key(p) -> Tuple[float, float]:
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def _spread(P: np.ndarray) -> float:
    return float(np.max(np.ptp(P, axis=0))) if len(P) > 1 else 0.0


def _estimate(ctx: Ctx) -> Tuple[Optional[np.ndarray], float]:
    """The best-fitting centre's neighbourhood, weighted by how well each centre explains the stomps."""
    m = _cup_map(ctx)
    best = int(np.argmin(m["cost"]))
    near = np.linalg.norm(m["C"] - m["C"][best], axis=1) < 0.012
    ok = near & (m["cost"] <= float(m["cost"][best]) + FIT_TOL)
    w = np.exp(-(m["cost"][ok] - float(m["cost"][best])) / 0.0008)
    return (m["C"][ok] * w[:, None]).sum(axis=0) / w.sum(), float(m["cost"][best])


def _localize(ctx: Ctx, arm: str):
    """After the first hit, stomp where the expected spread of the centres still in play shrinks most,
    until they agree to a few millimetres."""
    tried = set()
    for _ in range(18):
        P = _possible(ctx, FIT_TOL)
        if len(P) == 0:
            return None
        spread = _spread(P)
        if spread <= 0.006:
            break
        c = P.mean(axis=0)
        sub = P[:: max(1, len(P) // 250)]
        ranked = []
        for dx in np.arange(-0.080, 0.0801, 0.010):
            for dy in np.arange(-0.080, 0.0801, 0.010):
                p = c + np.array([dx, dy])
                if _key(p) in tried or not _scan_ok(p) or not _in_half(arm, p):
                    continue
                ms = _predict(ctx, arm, p, sub)
                hit = ms > 0
                frac = float(np.mean(hit))
                if frac <= 0.0 or frac >= 1.0:
                    continue
                expect = frac * _spread(sub[hit]) + (1.0 - frac) * _spread(sub[~hit])
                ranked.append((expect + 0.01 * float(np.mean(np.abs(ms) < 0.0015)), p))
        if not ranked:
            ctx.event(f"no stomp left that splits the {spread * 1000:.0f} mm of centres still in play")
            break
        ranked.sort(key=lambda t: t[0])
        # Score the short list again with the pose each stomp will really take: the yaw the planner settles on
        # turns the 64 mm pad pair about the fingertip point and changes which rim it can land on.
        real = []
        width = float(ctx.width(arm))
        for _score, p in ranked[:24]:
            solved = _solve_stomp(ctx, arm, p)
            if solved is None:
                tried.add(_key(p))
                continue
            ms = _rim_margin(_pad_footprints(solved[1], width), sub)
            hit = ms > 0
            frac = float(np.mean(hit))
            if frac <= 0.0 or frac >= 1.0:
                tried.add(_key(p))
                continue
            real.append((frac * _spread(sub[hit]) + (1.0 - frac) * _spread(sub[~hit])
                         + 0.01 * float(np.mean(np.abs(ms) < 0.0015)), p))
        real.sort(key=lambda t: t[0])
        moved = False
        for _score, p in real[:6]:
            tried.add(_key(p))
            r = yield from _s_stomp(ctx, arm, p)
            if r.get("reachable"):
                moved = True
                break
        if not moved:
            ctx.event(f"none of the stomps that split the {spread * 1000:.0f} mm of centres can be reached")
            break
    centre, cost = _estimate(ctx)
    if centre is None or cost > 0.008:
        ctx.event("the stomps that stood up do not fit one cup rim; kept scanning")
        return None
    spread = _spread(_possible(ctx, FIT_TOL))
    ctx.event(f"rim fit: centre ({centre[0]:.3f}, {centre[1]:.3f}); centres that explain the stomps span "
              f"{spread * 1000:.0f} mm")
    return centre


def s_scan_half(ctx: Ctx, arm: str):
    """Stomp this arm's half with the pads flat. Left (cup) arm takes +y, right (tap) arm takes -y.

    A cell is skipped once no centre still in play would put a rim under the pads there."""
    if ctx.bel.found.get("stomp_cup") is not None:
        return {"msg": "the cup was already found on the other half"}
    centre = None
    done = 0
    todo = _stomp_grid(arm)
    for _pass in range(2):
        later = []
        for p in todo:
            if not np.any(_predict(ctx, arm, p, _possible(ctx)) > 0.0015):
                continue
            r = yield from _s_stomp(ctx, arm, p)
            if not r.get("reachable"):
                later.append(p)
                continue
            done += 1
            if not r.get("occupied"):
                continue
            ctx.event(f"{arm} flat stomp hit at ({p[0]:.3f}, {p[1]:.3f}) after {done}, pads stopped at {r['z'] * 100:.1f} cm")
            centre = yield from _localize(ctx, arm)
            if centre is not None:
                break
        if centre is not None or not later:
            break
        todo = later
    side = "left" if arm == "cup" else "right"
    if centre is None:
        yield from _fold_clear(ctx, arm)
        return {"msg": f"nothing standing on the {side} half after {done} flat stomps"}
    ctx.bel.found["stomp_cup"] = np.asarray(centre, float).copy()
    ctx.bel.cup_xy = np.asarray(centre, float).copy()
    ctx.scratch["fit_xy"] = np.asarray(centre, float).copy()
    ctx.scratch["wide_search"] = True
    ctx.event(f"flat-hand centre ({centre[0]:.3f}, {centre[1]:.3f})")
    yield from _fold_clear(ctx, arm)
    return {"msg": f"cup under the flat {side} hand at ({centre[0]:.3f}, {centre[1]:.3f}) after {done} stomps"}


def s_scan_right(ctx: Ctx):
    res = yield from s_scan_half(ctx, "tap")
    if ctx.bel.found.get("stomp_cup") is None:
        raise SkillFault("both arms stomped their halves of the mat and nothing stood up")
    return res

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
    pads = ctx.scratch.get("probe") == "stamp"
    v = 0
    for pk in ctx.scratch.get("peck_log", []):
        if pk["occupied"]:
            if not P.cup_hits_arm("cup", pk["q_post"], pk["grip"], c, pads_only=pads):
                v += 1
            if P.cup_hits_arm("cup", pk["q_pre"], pk["grip"], c, pads_only=pads):
                v += 1
        else:
            if (P.cup_hits_arm("cup", pk["q_stop"], pk["grip"], c, pads_only=pads)
                    or P.cup_hits_arm("cup", pk["q_bottom"], pk["grip"], c, pads_only=pads)):
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
    region = feasible_region(cand)
    ctx.scratch["feasible_radius_m"] = region["radius_m"]
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
            ctx.event(f"coarse hit at ({p[0]:.3f}, {p[1]:.3f}): stamp stopped at {r['z'] * 100:.1f} cm after {done} pecks")
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


# Just outside the valve. One centimetre from the centre is still on the knob (the valve is 21 mm across),
# so a close there feels metal in every direction and says nothing about the handle. The handle is the
# radius that sticks out past the knob; this is where a close can tell the line from empty air.
RAY_RADIUS = 0.024


def _ray_frame(pivot: np.ndarray, z: float, psi: float, radius: float = RAY_RADIUS) -> np.ndarray:
    """Hand on the circle, jaws closing across the radius, wrist outside the knob.

    The approach points toward the centre. The pads sit a few millimetres behind the TCP, so the TCP
    is set just inside the probe radius and the wrist stays out past the handle, clear of the tank.
    """
    outward = np.array([np.cos(psi), np.sin(psi), 0.0])
    inward = -outward
    j = np.array([-np.sin(psi), np.cos(psi), 0.0])
    p = np.array([float(pivot[0]), float(pivot[1]), z]) + outward * (radius - 0.008)
    return frame_from(inward, j, p)


def _angle_order(start: float, n: int = 18, step_deg: float = 20.0) -> list:
    step = np.radians(step_deg)
    out = [float(_wrap(start))]
    k = 1
    while len(out) < n:
        out.append(float(_wrap(start + k * step)))
        out.append(float(_wrap(start - k * step)))
        k += 1
    return out[:n]


def _back_off_handle(ctx: Ctx, psi: float, rev: bool, strict: bool):
    """Pull the open fingers back out along the pinch's own approach line."""
    kin = ctx.kin["tap"]
    T_g = lever_grip_frame(ctx.bel.lever_pivot, psi, reverse=rev)
    u = T_g[:3, 2]
    T0 = ctx.tcp("tap")
    T_pre = T0.copy()
    T_pre[:3, 3] = T0[:3, 3] - u * 0.055 + Z_UP * 0.012
    try:
        back = M.cart_move(kin, [T0, T_pre], ctx.q("tap"), v_cart=0.04, speed_scale=ctx.speed, min_T=0.4, max_joint_step=0.22)
        ctx.check_traj("tap", back, grip_width=ctx.width("tap"))
    except PlanError as exc:
        if strict:
            raise SkillFault(f"the hand has the handle but cannot back off it ({exc})")
        return
    yield from ctx.play("tap", back)


def _try_descend(ctx: Ctx, T: np.ndarray, width: float):
    """Come down onto T from above. A straight move from the knob clips the tank."""
    kin = ctx.kin["tap"]
    seeds = [ctx.q("tap"), np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.4, 1.4, 1.0, 0.6, 0.0, 0.0])]
    if not kin.ik_multi(T, seeds).ok:
        return False
        yield
    above = T.copy()
    above[2, 3] = float(T[2, 3]) + 0.08
    here = ctx.tcp("tap")
    up = here.copy()
    up[2, 3] = max(float(here[2, 3]) + 0.06, float(above[2, 3]))
    for poses in ([here, up, above, T], [here, above, T], [here, T]):
        try:
            traj = M.cart_move(kin, poses, ctx.q("tap"), v_cart=0.07, speed_scale=ctx.speed, min_T=0.45, max_joint_step=0.35)
            ctx.check_traj("tap", traj, grip_width=width)
        except PlanError:
            continue
        yield from ctx.play("tap", traj)
        return True
    return False


def _try_move(ctx: Ctx, T: np.ndarray, width: float):
    """Move the tap hand to T. Returns False when that pose, or the path to it, is not clear."""
    kin = ctx.kin["tap"]
    seeds = [ctx.q("tap"), np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.4, 1.4, 1.0, 0.6, 0.0, 0.0])]
    r = kin.ik_multi(T, seeds)
    if not r.ok:
        return False
        yield
    try:
        traj = M.cart_move(kin, [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.07, speed_scale=ctx.speed,
                           min_T=0.4, max_joint_step=0.35)
        ctx.check_traj("tap", traj, grip_width=width)
    except PlanError:
        try:
            traj = M.joint_move(ctx.q("tap"), r.q, speed_scale=ctx.speed, min_T=0.6)
            ctx.check_traj("tap", traj, grip_width=width)
        except PlanError:
            return False
    yield from ctx.play("tap", traj)
    return True


def _lift_reachable(ctx: Ctx, psi: float, rev: Optional[bool] = None) -> bool:
    """True when the pour pinch can stand beside a handle at this yaw, close on it, and lift it through
    the whole pour arc without a wrist flip. `rev` picks the forward or reverse pinch."""
    if rev is None:
        rev = bool(ctx.scratch.get("lever_rev", False))
    kin = ctx.kin["tap"]
    P = np.asarray(ctx.bel.lever_pivot, float)
    T_g = lever_grip_frame(P, psi, reverse=rev)
    u = T_g[:3, 2]
    T_pre = T_g.copy()
    T_pre[:3, 3] = T_g[:3, 3] - u * 0.055 + Z_UP * 0.012
    T_hi = T_pre.copy()
    T_hi[2, 3] += 0.07
    seeds = [ctx.q("tap")] + LEVER_SEEDS
    q = None
    for T in (T_hi, T_pre, T_g):
        r = kin.ik_multi(T, ([q] if q is not None else []) + seeds)
        if not r.ok:
            return False
        q = r.q
    for th in np.linspace(0.0, LEVER_OPEN_THETA + 0.05, 12)[1:]:
        r = kin.ik_multi(lever_grip_frame(P, psi, th, reverse=rev), [q])
        if not r.ok or float(np.max(np.abs(r.q - q))) > 0.35:
            return False
        q = r.q
    return True


def _approach_yaw(ctx: Ctx, psi: float, rev: bool = False):
    """Stand beside a candidate radius the way the pour pinch does.

    The jaws stay near 4 cm on the way in. A 7 cm hover is used when the wrist can reach it, and a shorter
    standoff when the handle points to where that hover cannot. Fully open pads reach the tank.
    """
    kin = ctx.kin["tap"]
    seeds = [ctx.q("tap")] + LEVER_SEEDS
    width = 0.040
    T_g = lever_grip_frame(ctx.bel.lever_pivot, psi, reverse=rev)
    u = T_g[:3, 2]
    rg = kin.ik_multi(T_g, seeds)
    if not rg.ok:
        return False
        yield
    chosen = None
    for back in (0.055, 0.040, 0.028):
        T_pre = T_g.copy()
        T_pre[:3, 3] = T_g[:3, 3] - u * back + Z_UP * 0.010
        rp = kin.ik_multi(T_pre, [rg.q] + seeds)
        if not rp.ok:
            continue
        hops = [rp.q]
        T_hi = T_pre.copy()
        T_hi[2, 3] = float(T_pre[2, 3]) + 0.06
        rh = kin.ik_multi(T_hi, [rp.q] + seeds)
        if rh.ok:
            hops = [rh.q, rp.q]
        chosen = (T_pre, hops)
        break
    if chosen is None:
        ctx.event(f"radius {np.degrees(psi):.0f} deg: no standoff")
        return False
    T_pre, hops = chosen
    if abs(ctx.width("tap") - width) > 0.008:
        yield from ctx.grip_open("tap", width, cap=8.0)
    # Joint moves dip through the handle and turn it. Travel at a constant height above the valve, then come down.
    moved = False
    last = "no path"
    overhead = T_pre.copy()
    overhead[2, 3] = max(float(T_pre[2, 3]) + 0.08, 0.30)
    high = ctx.tcp("tap").copy()
    high[2, 3] = float(overhead[2, 3])
    try:
        over = M.cart_move(kin, [ctx.tcp("tap"), high, overhead, T_pre], ctx.q("tap"), v_cart=0.16,
                           speed_scale=ctx.speed, min_T=0.40, max_joint_step=0.35, timing="travel")
        if over.duration > 12.0:
            raise PlanError(f"overhead transit would take {over.duration:.0f}s")
        ctx.check_traj("tap", over, grip_width=width)
        slide = M.cart_move(kin, [T_pre, T_g], hops[-1], v_cart=0.04, speed_scale=ctx.speed,
                            min_T=0.35, max_joint_step=0.22)
        ctx.check_traj("tap", slide, grip_width=width)
        yield from ctx.play("tap", over)
        yield from ctx.play("tap", slide)
        moved = True
    except PlanError as exc:
        last = str(exc)
    if not moved:
        # A joint-space hop dips through the handle and turns it. Skip the radius instead.
        ctx.event(f"radius {np.degrees(psi):.0f} deg path: {last[:80]}")
    return moved


# The tape holds the dispenser until about 9 N. A push into the pivot below this does not slide it.
# The handle's own yaw friction is 0.05 N·m, about 2 N at the grip, so a 3 N cap still turns the handle.
INWARD_LIMIT_N = 3.0
PUSH_CAP = np.radians(18.0)


def _inward_n(ctx: Ctx) -> float:
    """How hard the fingertip is pressing the valve toward its centre, from the torque residual.

    Positive means the valve is pushing the hand outward, which is the hand pressing the dispenser inward.
    The skill never reads the dispenser's true pose.
    """
    r = ctx.mon["tap"].tcp[:3, 3] - np.asarray(ctx.bel.lever_pivot, float)
    r[2] = 0.0
    nrm = float(np.linalg.norm(r))
    if nrm < 1e-4:
        return 0.0
    return float(np.dot(ctx.mon["tap"].force, r / nrm))


def _retreat_radial(ctx: Ctx, dist: float = 0.008):
    """Step the fingertip away from the pivot so a shove into the valve does not keep loading the tape."""
    pivot = np.asarray(ctx.bel.lever_pivot, float)
    r = ctx.tcp("tap")[:3, 3] - pivot
    r[2] = 0.0
    nrm = float(np.linalg.norm(r))
    if nrm < 1e-4:
        return
    T = ctx.tcp("tap").copy()
    T[:3, 3] = T[:3, 3] + (r / nrm) * dist
    T[2, 3] += 0.012
    try:
        tr = M.cart_move(ctx.kin["tap"], [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.04, speed_scale=ctx.speed,
                         min_T=0.3, max_joint_step=0.25)
        yield from ctx.play("tap", tr)
    except PlanError:
        pass


def _yaw_held_to_park(ctx: Ctx, pivot: np.ndarray, yaw: float, goal: Optional[float] = None) -> float:
    """Turn a handle the jaws are already holding, about the knob, until it faces the pour.

    Releasing and re-closing at 45 degrees misses, and a stiff fingertip arc drags the taped dispenser.
    The grip point stays on the handle's circle. If the residual says the hand is bearing into the pivot,
    or the forearm would meet the tank, the turn stops where it is.
    """
    goal = float(PARK_YAW if goal is None else goal)
    kin = ctx.kin["tap"]
    ctx.monitor("tap", True)
    w0 = float(ctx.width("tap"))
    pivot = np.asarray(pivot, float)
    fails = 0
    for _ in range(14):
        err = _wrap(goal - yaw)
        if abs(err) <= np.radians(4.0):
            break
        step = float(np.clip(err, -np.radians(8.0), np.radians(8.0)))
        nxt = float(_wrap(yaw + step))
        T = lever_frame(ctx, nxt)
        try:
            traj = M.cart_move(kin, [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.018, speed_scale=ctx.speed,
                               min_T=0.4, max_joint_step=0.18, step_m=0.003)
            ctx.check_traj("tap", traj, grip_width=w0)
        except PlanError as exc:
            fails += 1
            ctx.event(f"that step of the turn is blocked ({exc})")
            if fails >= 2:
                break
            continue
        # The squeeze on the handle already shows a few newtons. Only a harder push, still under the
        # 9 N tape, means the wrist is bearing into the valve rather than turning the handle.
        shoved = {"n": 0}

        def guard(shoved=shoved) -> bool:
            if _inward_n(ctx) > 3.5:
                shoved["n"] += 1
            else:
                shoved["n"] = 0
            return shoved["n"] >= 5

        done = yield from ctx.play("tap", traj, guard=guard)
        if ctx.width("tap") > w0 + 0.004:
            ctx.event("the handle slipped out of the fingers while turning")
            break
        if not done:
            fails += 1
            ctx.event("eased off the valve and continued the turn")
            yield from _retreat_radial(ctx, 0.005)
            if fails >= 3:
                break
            continue
        fails = 0
        yaw = nxt
    ctx.event(f"held handle turned to {np.degrees(_wrap(yaw)):.0f} deg")
    return float(_wrap(yaw))


def _jaw_kind(res) -> str:
    """A square pinch on the 15 mm paddle stalls at 15.0 mm. Ten degrees off stalls near 17 mm.

    A stall near the 6 mm thickness is the paddle's edge: the handle is a few tens of degrees off, and it is not a miss.
    """
    if not res or not res.get("stalled") or float(res.get("force", 0.0)) <= 1.0:
        return "clear"
    w = float(res["width"])
    if 0.0142 < w < 0.0158:
        return "square"
    if 0.0120 < w < 0.0205:
        return "warm"
    if 0.0055 <= w <= 0.0120:
        return "edge"
    return "clear"


def _blade_pose(pivot: np.ndarray, theta: float) -> np.ndarray:
    """Closed finger, approach along the circle, so a push turns the handle instead of prying the valve inward.

    The contact is out near the paddle tip. In closer, the fingertip meets the tank.
    """
    radial = np.array([np.cos(theta), np.sin(theta), 0.0])
    tangent = np.array([-np.sin(theta), np.cos(theta), 0.0])
    p = np.asarray(pivot, float) + radial * 0.040 + tangent * 0.006 + Z_UP * (LEVER_GRIP_Z + 0.004)
    return frame_from(tangent, Z_UP, p)


def _play_push(ctx: Ctx, poses: list) -> bool:
    kin = ctx.kin["tap"]
    seeds = [ctx.q("tap")] + LEVER_SEEDS
    T = poses[0]
    up = T.copy()
    up[2, 3] = float(T[2, 3]) + 0.05
    rh = kin.ik_multi(up, seeds)
    rg = kin.ik_multi(T, seeds)
    if not (rh.ok or rg.ok):
        return False
    q = rh.q if rh.ok else rg.q
    tr = M.via_move([ctx.q("tap"), q], speed_scale=ctx.speed, min_T=0.6)
    ctx.check_traj("tap", tr, grip_width=0.0)
    yield from ctx.play("tap", tr)
    if rh.ok:
        down = M.cart_move(kin, [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.04, speed_scale=ctx.speed, min_T=0.3, max_joint_step=0.25)
        ctx.check_traj("tap", down, grip_width=0.0)
        yield from ctx.play("tap", down)
    if len(poses) > 1:
        arc = [ctx.tcp("tap")] + list(poses[1:])
        mv = M.cart_move(kin, arc, ctx.q("tap"), v_cart=0.01, speed_scale=ctx.speed, min_T=0.5, step_m=0.002, max_joint_step=0.22)
        ctx.check_traj("tap", mv, grip_width=0.0)
        yield from ctx.play("tap", mv)
    return True


def _blade_push(ctx: Ctx, yaw: float, direction: float, span: float) -> bool:
    pivot = np.asarray(ctx.bel.lever_pivot, float)
    yield from ctx.grip_open("tap", 0.0, cap=8.0, tol=0.002, timeout=2.2)
    start = float(yaw - direction * np.radians(12.0))
    for use in (span, span * 0.55):
        end = float(yaw + direction * use)
        n = max(int(abs(end - start) / 0.07), 2)
        poses = [_blade_pose(pivot, th) for th in np.linspace(start, end, n + 1)]
        try:
            if (yield from _play_push(ctx, poses)):
                return True
        except PlanError as exc:
            ctx.event(f"blade push blocked ({exc})")
    return False


def _sweep_push(ctx: Ctx, yaw: float, direction: float, span: float) -> bool:
    """The closed-finger pose of the pour pinch, walked a short way around the valve. Reachable on the whole circle."""
    pivot = np.asarray(ctx.bel.lever_pivot, float)
    tool = lever_grip_frame(pivot, float(PARK_YAW))[:3, :3]
    yield from ctx.grip_open("tap", 0.0, cap=8.0, tol=0.002, timeout=2.2)
    start = float(yaw - direction * np.radians(14.0))
    end = float(yaw + direction * span)
    n = max(int(abs(end - start) / 0.08), 2)
    poses = [_sweep_pose(ctx, th, tool) for th in np.linspace(start, end, n + 1)]
    try:
        return bool((yield from _play_push(ctx, poses)))
    except PlanError as exc:
        ctx.event(f"short push blocked ({exc})")
        return False


def _leave_handle(ctx: Ctx, yaw: float, rev: bool):
    yield from ctx.grip_open("tap", 0.040, cap=6.0)
    yield from _back_off_handle(ctx, yaw, rev, strict=False)
    if float(np.linalg.norm(ctx.tcp("tap")[:3, 3] - np.asarray(ctx.bel.lever_pivot, float))) < 0.05:
        T = ctx.tcp("tap").copy()
        T[2, 3] = float(T[2, 3]) + 0.06
        try:
            up = M.cart_move(ctx.kin["tap"], [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.05, speed_scale=ctx.speed, min_T=0.3, max_joint_step=0.25)
            ctx.check_traj("tap", up, grip_width=ctx.width("tap"))
            yield from ctx.play("tap", up)
        except PlanError:
            pass


def _refine_square(ctx: Ctx, psi: float, rev: bool):
    """The jaws touched the paddle but were not square on it. Step around that radius until the stall is 15 mm."""
    for deg in (7.0, -7.0, 12.0, -12.0, 4.0, -4.0):
        ang = float(_wrap(psi + np.radians(deg)))
        if not (yield from _approach_yaw(ctx, ang, rev)):
            continue
        res = yield from ctx.grip_close("tap", 6.0, rate=0.035)
        kind = _jaw_kind(res)
        ctx.event(f"refine {np.degrees(ang):.0f} deg: jaws {fmt_mm(res['width'])}, {kind}")
        if kind == "square":
            return ang, rev, res
        yield from ctx.grip_open("tap", 0.040, cap=8.0)
        yield from _back_off_handle(ctx, ang, rev, strict=False)
    return None


def _push_then_grasp(ctx: Ctx, yaw: float, direction: float) -> float:
    """Move a handle the pinch cannot lift across the short gap, then take the pour pinch and finish the turn."""
    ctx.event(f"pushing the handle from {np.degrees(yaw):.0f} deg toward the pour")
    # The tangent blade meets the tank on the far side of the valve. The pour pinch, walked a short
    # arc, stays outside it. A long chase is what slides the taped dispenser.
    pushed = yield from _sweep_push(ctx, yaw, direction, np.radians(36.0))
    if not pushed:
        pushed = yield from _blade_push(ctx, yaw, direction, np.radians(32.0))
    ctx.scratch["lever_rev"] = False
    for extra in (28.0, 18.0, 40.0, 10.0, 52.0):
        ang = float(_wrap(yaw + direction * np.radians(extra)))
        if not (yield from _approach_yaw(ctx, ang, False)):
            continue
        res = yield from ctx.grip_close("tap", 6.0, rate=0.035)
        if _jaw_kind(res) == "square":
            ctx.scratch["lever_rev"] = False
            yaw2 = float((yield from _yaw_held_to_park(ctx, np.asarray(ctx.bel.lever_pivot, float), ang, float(PARK_YAW))))
            yield from _leave_handle(ctx, yaw2, False)
            ctx.bel.lever_yaw = float(_wrap(yaw2))
            ctx.event(f"handle parked at {np.degrees(yaw2):.0f} deg")
            return float(_wrap(yaw2))
        yield from ctx.grip_open("tap", 0.040, cap=8.0)
        yield from _back_off_handle(ctx, ang, False, strict=False)
    raise SkillFault(f"felt the handle near {np.degrees(yaw):.0f} deg but could not bring it to the pour angle")


def _settle_held(ctx: Ctx, psi: float, rev: bool) -> float:
    """The jaws are square on the paddle. Turn it to the pour angle, or push across the gap the wrist cannot follow."""
    ctx.scratch["lever_rev"] = bool(rev)
    yaw = float((yield from _yaw_held_to_park(ctx, np.asarray(ctx.bel.lever_pivot, float), float(psi), float(PARK_YAW))))
    near = abs(_wrap(yaw - float(PARK_YAW))) <= np.radians(12.0)
    if near or (not rev and _lift_reachable(ctx, yaw, False)):
        yield from _leave_handle(ctx, yaw, rev)
        ctx.scratch["lever_rev"] = bool(rev)
        ctx.bel.lever_yaw = float(_wrap(yaw))
        return float(_wrap(yaw))
    direction = float(np.sign(_wrap(float(PARK_YAW) - yaw)) or 1.0)
    yield from ctx.grip_open("tap", 0.040, cap=6.0)
    return (yield from _push_then_grasp(ctx, yaw, direction))


def s_pinch_lever(ctx: Ctx):
    """Find the handle from above.

    The handle is a radius of the knob. A close that stalls at 15.0 mm has the paddle square in the jaws, and
    that radius is the handle's angle. A close a few degrees off stalls wider and is not trusted. When the
    wrist cannot lift the angle it found, the jaws turn the handle to the pour; a short push crosses the gap
    the pinch cannot reach. The dispenser is not swept.
    """
    # Forward pinches only where the wrist can lift. The south and north holes are felt with the reverse
    # pinch, whose edge stall is a real touch. Forward closes in the hole sweep the paddle and turn it.
    forward = [np.radians(d) for d in (45, 30, 60, 15, 75, 0, -10)]
    reverse = [np.radians(d) for d in (-95, -105, -115, -130, -150, -170, 170, 150, 130, 115, 100)]
    found = None
    tried = 0
    for psi, rev in [(a, False) for a in forward] + [(a, True) for a in reverse]:
        if not (yield from _approach_yaw(ctx, psi, rev)):
            continue
        tried += 1
        res = yield from ctx.grip_close("tap", 6.0, rate=0.035)
        kind = _jaw_kind(res)
        pinch = "reverse" if rev else "forward"
        ctx.event(f"radius {np.degrees(psi):.0f} deg ({pinch}): jaws {fmt_mm(res['width'])}, {res['force']:.1f} N, {kind}")
        if kind == "square":
            found = (float(psi), rev, res)
            break
        if kind == "edge":
            # A stall on the 6 mm edge will not become a 15 mm square a few degrees away.
            # Push the handle toward the pour instead of closing on more empty radii.
            yield from ctx.grip_open("tap", 0.040, cap=8.0)
            yield from _back_off_handle(ctx, psi, rev, strict=False)
            found = ("push", float(psi), rev)
            break
        if kind == "warm":
            yield from ctx.grip_open("tap", 0.040, cap=8.0)
            yield from _back_off_handle(ctx, psi, rev, strict=False)
            refined = yield from _refine_square(ctx, float(psi), rev)
            found = refined if refined is not None else ("push", float(psi), rev)
            break
        yield from ctx.grip_open("tap", 0.040, cap=8.0)
        yield from _back_off_handle(ctx, psi, rev, strict=False)
    if found is None:
        ctx.monitor("tap", False)
        raise SkillFault(f"closed on {tried} radii around the knob and never held the handle")
    if found[0] == "push":
        direction = float(np.sign(_wrap(float(PARK_YAW) - found[1])) or 1.0)
        yaw = yield from _push_then_grasp(ctx, float(found[1]), direction)
        res_w = 0.015
        psi = found[1]
    else:
        psi, rev, res = found
        res_w = res["width"]
        yaw = yield from _settle_held(ctx, float(psi), bool(rev))
    ctx.monitor("tap", False)
    ctx.bel.found["lever_yaw_before"] = float(_wrap(float(psi)))
    ctx.bel.lever_yaw = float(yaw)
    return {"msg": (f"handle squared at {np.degrees(psi):.0f} deg, jaws {fmt_mm(res_w)}; "
                    f"left at {np.degrees(yaw):.0f} deg")}


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
            # An inward residual means the finger is bearing on the valve, not riding the handle's circle.
            # Stop before that force reaches the tape (9 N). The tangent test still needs a short run.
            inward = _inward_n(ctx) > INWARD_LIMIT_N
            if push > thr or inward:
                st["n"] += 1
            else:
                st["n"] = 0
            if st["n"] >= (3 if inward else 6):
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

    # --- 2. nudge it toward the pour angle. A short tangential push, never a full turn.
    # The old arc took (goal - now) mod 2π, which is almost a revolution when the touch estimate wraps,
    # and that spin is what dragged the dispenser (seed-level pushes of 10 N through the paddle).
    for attempt in range(3):
        err = _wrap(yaw - PARK_YAW)
        if abs(err) <= PARK_TOLERANCE:
            break
        prev_abs = abs(err)
        want = 1.0 if err < 0 else -1.0
        if want != side:
            far = yaw - want * np.radians(40.0)
            far = yield from go_above(far)
            ctx.monitor("tap", True)
            th_c, _ = yield from sweep(far, far + want * np.radians(70.0), want)
            if th_c is None:
                ctx.event("lost the handle while coming round to the other side; leaving it where it was last felt")
                break
            y2 = touch_yaw(ctx, ctx.mon["tap"].tcp, ctx.width("tap"), want)
            if y2 is not None:
                yaw = _wrap(y2)
            side = want
            ctx.event(f"touched the handle from the other side: it lies at about {np.degrees(yaw):.0f} deg")
            if abs(_wrap(yaw - PARK_YAW)) <= PARK_TOLERANCE:
                break
        # Follow the handle's own circle, and only a little way. Re-measure before pushing again.
        span = min(abs(_wrap(yaw - PARK_YAW)) + float(PARK_OVERSHOOT), PUSH_CAP)
        th_now = finger_angle()
        th_goal = th_now + want * span
        poses = [ctx.tcp("tap")] + arc_poses(th_now, th_goal, 0.08)
        try:
            tr = M.cart_move(kin, poses, ctx.q("tap"), v_cart=0.015, speed_scale=ctx.speed, min_T=0.8, step_m=0.003, max_joint_step=0.18)
            ctx.check_traj("tap", tr, grip_width=0.0)
        except PlanError as exc:
            ctx.event(f"the short push would hit something ({exc}); leaving the handle")
            break
        shoved = {"n": 0}

        def push_guard(shoved=shoved) -> bool:
            if _inward_n(ctx) > INWARD_LIMIT_N:
                shoved["n"] += 1
            else:
                shoved["n"] = 0
            return shoved["n"] >= 3

        done = yield from ctx.play("tap", tr, guard=push_guard)
        if not done:
            ctx.event("fingertip push would move the dispenser; backing off")
            yield from _retreat_radial(ctx)
            break
        yield from ctx.settle("tap", 0.0015, 2.0)
        y_v = yield from probe(want, finger_angle())
        if y_v is None:
            ctx.event("could not feel the handle again after the short push; not starting another lap")
            break
        yaw = _wrap(y_v)
        ctx.event(f"touched it again: the handle now sits at {np.degrees(yaw):.1f} deg")
        if abs(_wrap(yaw - PARK_YAW)) > prev_abs - np.radians(3.0):
            ctx.event("the push did not bring the handle closer to the pour angle; stopping")
            break
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


def s_lever_search(ctx: Ctx):
    """Find the handle by closing on it. A fingertip sweep around the valve is what dragged the taped dispenser,
    so a handle the jaws cannot hold is left where it is rather than pushed."""
    return (yield from s_pinch_lever(ctx))
