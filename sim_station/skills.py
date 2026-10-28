"""Manipulation skills. Every skill is a generator: each `yield` is one 2 ms control tick for that arm's lane.

Skills use only honest information: encoder angles (forward kinematics), motor torque feedback (contact monitor) and
gripper width/effort. They never read object poses from the simulator. Where a measurement is needed (jaw stall width,
contact height, handle angle), the skill measures it and raises `SkillFault` with the sensed numbers if it is wrong.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.kin import frame_from, rot_about, rotz
from sim_station.motion import PlanError
from sim_station.runtime import Ctx, SkillFault, CTRL_STEPS
from sim_station.sensing import Baseline

HOLD_DZ = L.CUP_GRASP_HEIGHT_ABOVE_BASE      # TCP height above the cup base when held at mid height
Z_UP = np.array([0.0, 0.0, 1.0])
LEVER_TILT_DEG = 40.0                          # pinch approach, below horizontal, along the handle
LEVER_OPEN_THETA = 1.05                        # how far the handle is lifted (flow starts at 0.45)
LEVER_GRIP_Z = 0.0015 + 0.5 * L.LEVER_HANDLE_THICK


def side_frame(kin, xy, z, off=-9.0, tilt=0.0, depth=0.0) -> np.ndarray:
    base = kin.mount[:2]
    az = np.arctan2(xy[1] - base[1], xy[0] - base[0]) + np.radians(off)
    t = np.radians(tilt)
    a = np.array([np.cos(az) * np.cos(t), np.sin(az) * np.cos(t), np.sin(t)])
    j = np.array([-np.sin(az), np.cos(az), 0.0])
    return frame_from(a, j, np.array([xy[0], xy[1], z]) + a * depth)


def fmt_mm(x: float) -> str:
    return f"{x * 1000:.1f} mm"


def cap_for(ctx: Ctx) -> float:
    return L.FORCE_CAP_PLASTIC_CUP_N if ctx.plant.material == "plastic" else L.FORCE_CAP_PAPER_CUP_N


# ---------------------------------------------------------------------------
# generic
# ---------------------------------------------------------------------------

def s_wait(ctx: Ctx, seconds: float):
    yield from ctx.wait(seconds)
    return {"msg": "still"}


def s_open(ctx: Ctx, arm: str, width: float = L.GRIPPER_MAX_OPEN_M):
    w = yield from ctx.grip_open(arm, width)
    return {"msg": f"jaws open to {fmt_mm(w)}"}


def s_goto_rest(ctx: Ctx, arm: str):
    q0 = ctx.q(arm)
    traj = M.via_move([q0, L.REST_Q], speed_scale=ctx.speed, min_T=0.8)
    ctx.check_traj(arm, traj)
    yield from ctx.play(arm, traj)
    ctx.q_cmd[arm] = L.REST_Q.copy()
    return {"msg": "folded at rest"}


def s_goto_pose(ctx: Ctx, arm: str, T: np.ndarray, via_up: float = 0.0, v: float = L.TRANSIT_SPEED_M_S, label: str = "pose"):
    kin = ctx.kin[arm]
    q0 = ctx.q(arm)
    r = kin.ik_multi(T, [q0, np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0])])
    if not r.ok:
        raise SkillFault(f"{label} unreachable: {r.reason}")
    qs = [q0]
    if via_up > 0.0:
        Tv = T.copy()
        Tv[2, 3] += via_up
        rv = kin.ik_multi(Tv, [q0, r.q])
        if rv.ok:
            qs.append(rv.q)
    qs.append(r.q)
    traj = M.via_move(qs, speed_scale=ctx.speed)
    ctx.check_traj(arm, traj)
    yield from ctx.play(arm, traj)
    return {"msg": f"{label} reached"}


# ---------------------------------------------------------------------------
# cup arm: pick an upright cup from the side
# ---------------------------------------------------------------------------

def s_approach_cup(ctx: Ctx, xy: np.ndarray, base_z: float):
    z = base_z + HOLD_DZ
    az = ctx.scratch.pop("regrasp_az", None)
    fr = ctx.side_frames("cup", xy, z, depth=0.004, standoff=0.05, az_fixed=az)
    ctx.scratch["cup_fr"] = fr
    q0 = ctx.q("cup")
    traj = M.via_move([q0, fr["q"]["hi"], fr["q"]["pre"]], speed_scale=ctx.speed)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    return {"msg": f"standoff 5 cm in front of the cup, approach tilt {fr['tilt']:.0f} deg"}


def cup_radius_at_hold() -> float:
    """Half the width the jaws stall at when the hand is on the cup's axis (measured once on the calibration cup)."""
    return 0.5 * L.CUP_HOLD_WIDTH_ON_AXIS


def s_slide_in(ctx: Ctx):
    """Slide the open fingers in around the cup wall. If a fingertip lands on the rim, comply sideways (admittance on the
    lateral contact force) so the lead-in funnel can centre the hand on the cup instead of pushing the cup away."""
    fr = ctx.scratch["cup_fr"]
    kin = ctx.kin["cup"]
    q0 = ctx.q("cup")
    traj = M.cart_move(kin, [ctx.tcp("cup"), fr["grasp"]], q0, v_cart=L.APPROACH_SPEED_M_S, speed_scale=ctx.speed)
    ctx.monitor("cup", True)
    a, j = fr["a"], fr["j"]
    j_base = kin.R_wb.T @ j
    delta = 0.0
    dt = CTRL_STEPS * ctx.plant.dt
    jammed = False
    for k_ in range(len(traj)):
        m = ctx.mon["cup"]
        if float(-(m.force @ a)) > 5.0:
            jammed = True
            break
        f_j = float(m.force @ j)
        if abs(f_j) > 0.35:
            delta = float(np.clip(delta + 0.035 * (f_j - np.sign(f_j) * 0.35) * dt, -0.022, 0.022))
        q_k = traj.q[k_]
        if delta != 0.0:
            Jp = kin.jacobian(q_k)[:3]
            q_k = q_k + np.linalg.pinv(Jp, rcond=1e-3) @ (j_base * delta)
        ctx.q_cmd["cup"] = q_k
        ctx.plant.command_arm("cup", q_k, traj.qd[k_])
        yield
    ctx.monitor("cup", False)
    if jammed:
        ctx.q_cmd["cup"] = ctx.q("cup")
        raise SkillFault("fingers jammed on the cup rim while sliding in (contact force above 5 N)")
    if delta != 0.0:
        fr["grasp"] = fr["grasp"].copy()
        fr["grasp"][:3, 3] = fr["grasp"][:3, 3] + j * delta
        ctx.event(f"fingertip met the rim: complied {1000 * delta:+.0f} mm sideways to centre on the cup")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    return {"msg": "fingers straddle the cup wall" + (f" (centred {1000 * delta:+.0f} mm by touch)" if delta != 0.0 else "")}


CHORD_ACCEPT_S = 0.004          # accept a close whose chord puts the hand within 5 mm of the cup's axis


def s_close_cup(ctx: Ctx):
    """Close on the cup and servo the hand onto its axis from the width the jaws stall at: a close on the axis stalls at the diameter,
    off it at a shorter chord 2 sqrt(r^2 - s^2). The chord gives |s| but not the sign, so move s deeper and look again; if the chord got
    shorter we went the wrong way. A close on nothing falls back to a small search around the believed position."""
    cap = cap_for(ctx)
    kin = ctx.kin["cup"]
    fr = ctx.scratch["cup_fr"]
    a, j = fr["a"], fr["j"]
    r_c = cup_radius_at_hold()
    fallback = [(0.014, 0.0), (-0.014, 0.0), (0.0, 0.016), (0.014, 0.016), (-0.014, 0.016), (0.0, -0.012)]
    lat, deep = 0.0, 0.0
    sign, s_prev, n = 1.0, None, 0
    last = ""
    fb = 0
    while n < 9:
        res = yield from ctx.grip_close("cup", cap, rate=0.05)
        w = res["width"]
        s = float(np.sqrt(max(r_c * r_c - (0.5 * w) ** 2, 0.0))) if res["stalled"] else None
        if res["stalled"] and 0.050 < w < 0.075 and s <= CHORD_ACCEPT_S:
            ctx.bel.cup_in_hand = True
            ctx.scratch["hold_width"] = w
            extra = f" after {n} re-try(ies)" if n else ""
            return {"msg": f"jaws stalled at {fmt_mm(w)} with {res['force']:.1f} N of {cap:.0f} N cap: cup in hand{extra}"}
        n += 1
        yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M)
        if res["stalled"] and 0.025 < w < 0.075:
            if s_prev is not None and s > s_prev + 0.002:
                sign = -sign
            s_prev = s
            last = f"stalled on a {1000 * w:.0f} mm chord"
            ctx.event(f"{last}: the hand is about {1000 * s:.0f} mm off the cup's axis; moving {sign * 1000 * s:+.0f} mm along the approach")
            deep += sign * s
        else:
            last = "jaws closed on nothing"
            if fb >= len(fallback):
                break
            lat2, deep2 = fallback[fb]
            fb += 1
            ctx.event(f"{last}; shifting to {1000 * lat2:+.0f} mm across, {1000 * deep2:+.0f} mm deeper")
            lat, deep = lat2, deep2
            T0 = ctx.tcp("cup")
            T_back = T0.copy()
            T_back[:3, 3] = T0[:3, 3] - a * 0.045
            try:
                yield from ctx.play("cup", M.cart_move(kin, [T0, T_back], ctx.q("cup"), v_cart=0.06, speed_scale=ctx.speed, min_T=0.7))
            except PlanError:
                pass
        T_g = fr["grasp"].copy()
        T_g[:3, 3] = T_g[:3, 3] + j * lat + a * deep
        try:
            move = M.cart_move(kin, [ctx.tcp("cup"), T_g], ctx.q("cup"), v_cart=0.05, speed_scale=ctx.speed, min_T=0.7)
            yield from ctx.play("cup", move)
            yield from ctx.wait(L.CONTACT_PAUSE_S)
        except PlanError:
            last = "that re-try position is unreachable"
            if res["stalled"]:
                deep -= 2.0 * sign * s_prev if s_prev is not None else 0.0
                sign = -sign
    raise SkillFault(f"could not get the cup between the fingers ({last}) after {n} attempts")


def s_lift_cup(ctx: Ctx, dz: float = 0.05):
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    T1 = T0.copy()
    T1[2, 3] += dz
    traj = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.07, speed_scale=ctx.speed)
    w0 = ctx.width("cup")
    yield from ctx.play("cup", traj)
    if ctx.width("cup") > w0 + 0.0025:
        raise SkillFault("cup slipped in the jaws during the lift")
    return {"msg": f"lifted {dz * 100:.0f} cm, jaws hold {fmt_mm(ctx.width('cup'))}"}


# ---------------------------------------------------------------------------
# cup arm: spout and coaster
# ---------------------------------------------------------------------------

def spout_z() -> float:
    return L.MAT_SURFACE_Z + L.SPOUT_BASE_CLEARANCE + HOLD_DZ


def spout_entry(ctx: Ctx, T0: np.ndarray) -> np.ndarray:
    """Corridor entry point. Tries the side the cup comes from first, then the other, and for each entry distance and hand tilt
    checks the whole job: carry in, lower, slide under the outlet, with the remaining stack and the other arm as obstacles."""
    kin = ctx.kin["cup"]
    xs, ys = ctx.bel.spout_xy
    q0 = ctx.q("cup")
    seeds = [q0, np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0]), L.REAL_L1_CUP_WAYPOINTS["under_spout"][:6]]
    minus_x = [(np.array([xs - d, ys]), "-x") for d in (0.10, 0.12, 0.085, 0.14, 0.17)]
    plus_y = [(np.array([xs, ys + d]), "+y") for d in (0.17, 0.20, 0.14)]
    cands = (plus_y + minus_x) if T0[1, 3] > ys + 0.10 else (minus_x + plus_y)
    last = ""
    for e, side in cands:
        for tilt in (0.0, -12.0, -25.0, 12.0):
            try:
                ok = True
                for z in (0.118, spout_z() + 0.002, spout_z()):
                    if not kin.ik_multi(side_frame(kin, e, z, tilt=tilt), seeds).ok:
                        ok = False
                        last = f"entry ({e[0]:.3f}, {e[1]:.3f}) tilt {tilt:.0f}: unreachable"
                        break
                if not ok:
                    continue
                p1 = side_frame(kin, e, 0.118, tilt=tilt)
                p2 = side_frame(kin, e, spout_z() + 0.002, tilt=tilt)
                carry = M.cart_move(kin, [T0, p1, p2], q0, v_cart=L.CARRY_SPEED_M_S * 1.3, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.2)
                ctx.check_traj("cup", carry)
                slide = M.cart_move(kin, [p2, side_frame(kin, np.array([xs, ys]), spout_z(), tilt=tilt)], carry.q[-1], v_cart=0.10,
                                    speed_scale=ctx.speed, min_T=1.0, max_joint_step=0.2)
                ctx.check_traj("cup", slide)
            except PlanError as exc:
                last = f"entry ({e[0]:.3f}, {e[1]:.3f}) tilt {tilt:.0f}: {exc}"
                continue
            ctx.scratch["spout_entry"] = e
            ctx.scratch["spout_tilt"] = tilt
            ctx.scratch["spout_side"] = side
            return e
    raise PlanError(f"no collision-free corridor entry for the spout ({last})")


def s_carry_to_spout(ctx: Ctx):
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    e = spout_entry(ctx, T0)
    tilt = ctx.scratch["spout_tilt"]
    z_carry = 0.118
    p1 = side_frame(kin, e, z_carry, tilt=tilt)
    p2 = side_frame(kin, e, spout_z() + 0.002, tilt=tilt)
    traj = M.cart_move(kin, [T0, p1, p2], ctx.q("cup"), v_cart=L.CARRY_SPEED_M_S * 1.3, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.2)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    return {"msg": f"cup lowered to spout height in the corridor ({ctx.scratch['spout_side']} side of the dispenser)"}


def s_slide_under(ctx: Ctx):
    kin = ctx.kin["cup"]
    xs, ys = ctx.bel.spout_xy
    T0 = ctx.tcp("cup")
    T1 = side_frame(kin, np.array([xs, ys]), spout_z(), tilt=ctx.scratch.get("spout_tilt", 0.0))
    traj = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.10, speed_scale=ctx.speed, min_T=1.0, max_joint_step=0.2)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    yield from ctx.wait(0.25)
    ctx.scratch["under_spout_tcp"] = ctx.tcp("cup")[:3, 3].copy()
    return {"msg": "cup centred under the outlet, rim 1.5 cm below it"}


def s_exit_spout(ctx: Ctx):
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    e = ctx.scratch["spout_entry"]
    tilt = ctx.scratch.get("spout_tilt", 0.0)
    p1 = side_frame(kin, e, spout_z() + 0.002, tilt=tilt)
    p2 = side_frame(kin, e, 0.118, tilt=tilt)
    traj = M.cart_move(kin, [T0, p1, p2], ctx.q("cup"), v_cart=0.10, speed_scale=ctx.speed, min_T=1.0, max_joint_step=0.2)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    return {"msg": "slid out sideways under the outlet, then lifted clear"}


def s_to_coaster(ctx: Ctx, seat_z: float):
    kin = ctx.kin["cup"]
    cx, cy = ctx.bel.coaster_xy
    T0 = ctx.tcp("cup")
    seeds = [ctx.q("cup"), L.REAL_L1_CUP_WAYPOINTS["in_coaster"][:6], np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0])]
    chosen = None
    for tilt in (0.0, -10.0, -20.0, -30.0):
        top = side_frame(kin, np.array([cx, cy]), seat_z + HOLD_DZ + 0.035, tilt=tilt)
        low = side_frame(kin, np.array([cx, cy]), seat_z + HOLD_DZ - 0.006, tilt=tilt)
        if kin.ik_multi(top, seeds).ok and kin.ik_multi(low, seeds).ok:
            chosen = (tilt, top)
            break
    if chosen is None:
        raise SkillFault("the coaster cannot be reached with the cup upright")
    ctx.scratch["coaster_tilt"] = chosen[0]
    poses = [T0, chosen[1]]
    if ctx.bel.stack_n is not None:
        # the rest of the stack is still standing between the spout and the coaster: carry over it, not past it
        z_over = max(top_over(ctx) + 0.07, 0.25)
        up = T0.copy()
        up[2, 3] = z_over
        across = chosen[1].copy()
        across[2, 3] = z_over
        poses = [T0, up, across, chosen[1]]
    traj = M.cart_move(kin, poses, ctx.q("cup"), v_cart=L.CARRY_SPEED_M_S * 1.4, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.2)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    return {"msg": "cup held 3.5 cm above the coaster"}


def top_over(ctx: Ctx) -> float:
    """Height of the top of whatever is left of the stack, from the stack count the plan believes in."""
    n = max((ctx.bel.stack_n or 1) - 1, 1)
    return L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP + L.CUP_HEIGHT


def s_place_cup(ctx: Ctx, seat_z: float):
    """Guarded descent into the coaster funnel; stop on the contact force or at the expected seat."""
    kin = ctx.kin["cup"]
    cx, cy = ctx.bel.coaster_xy
    T0 = ctx.tcp("cup")
    z_goal = seat_z + HOLD_DZ - 0.004
    T1 = T0.copy()
    T1[2, 3] = z_goal
    plan = ctx.cart_repair("cup", [T0, T1], v_cart=0.018, min_T=1.0)
    for extra in plan[:-1]:
        yield from ctx.play("cup", extra)
    traj = plan[-1]
    ctx.monitor("cup", True)
    expected = seat_z + HOLD_DZ
    state = {"n": 0, "contact_z": None, "z_cmd": float(T0[2, 3]), "k": 0}

    def on_tick(k: int) -> None:
        state["k"] = k
        state["z_cmd"] = float(kin.fk(traj.q[k])[2, 3])

    def guard() -> bool:
        m = ctx.mon["cup"]
        z_now = float(m.tcp[2, 3])
        if z_now > expected + 0.016:                       # armed only within 16 mm of the believed seat
            return False
        # contact shows up two ways: a force step, or the hand lagging its command (the arm is a spring, 400 N/m at the TCP)
        lag = z_now - state["z_cmd"]
        if float(m.delta_force(60)[2]) > 1.0 or lag > 0.0045:
            state["n"] += 1
        else:
            state["n"] = 0
        if state["n"] >= 5:
            state["contact_z"] = z_now
            return True
        return False

    done = yield from ctx.play("cup", traj, guard=guard, on_tick=on_tick)
    ctx.monitor("cup", False)
    z_end = float(ctx.tcp("cup")[2, 3])
    expected = seat_z + HOLD_DZ
    if not done:
        return {"msg": f"cup base touched down at TCP z {z_end * 1000:.1f} mm (expected {expected * 1000:.1f} mm)"}
    return {"msg": f"cup lowered to the seat without a contact spike, TCP z {z_end * 1000:.1f} mm"}


def s_release(ctx: Ctx):
    yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M, cap=L.FORCE_CAP_PROBE_N)
    ctx.bel.cup_in_hand = False
    return {"msg": f"jaws opened to {fmt_mm(ctx.width('cup'))}, cup released"}


def s_retreat(ctx: Ctx, options=((0.05, 0.05), (0.035, 0.07), (0.02, 0.09), (0.0, 0.10))):
    """Pull the hand out of the served cup: back along the approach and up, taking the first reachable option."""
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    a_h = T0[:3, 2].copy()
    a_h[2] = 0.0
    a_h = a_h / max(np.linalg.norm(a_h), 1e-6)
    last = ""
    for back, up in options:
        T1 = T0.copy()
        T1[:3, 3] = T0[:3, 3] - a_h * back + Z_UP * up
        try:
            traj = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.8)
            ctx.check_traj("cup", traj)
        except PlanError as exc:
            last = str(exc)
            continue
        yield from ctx.play("cup", traj)
        return {"msg": f"hand backed out {back * 100:.0f} cm and raised {up * 100:.0f} cm clear of the served cup"}
    raise SkillFault(f"no reachable way to pull the hand out of the cup: {last}")


# ---------------------------------------------------------------------------
# tap arm: pinch the handle, lift it, hold, close it, let go
# ---------------------------------------------------------------------------

def lever_dirs(psi: float):
    d = np.array([np.cos(psi), np.sin(psi), 0.0])
    a = np.array([-np.sin(psi), np.cos(psi), 0.0])
    return d, a


def lever_grip_frame(pivot: np.ndarray, psi: float, theta: float = 0.0, tilt_deg: float = LEVER_TILT_DEG) -> np.ndarray:
    d, a = lever_dirs(psi)
    t = np.radians(tilt_deg)
    u = d * np.cos(t) - Z_UP * np.sin(t)
    G = pivot + d * L.LEVER_GRIP_DIST + Z_UP * LEVER_GRIP_Z
    T0 = frame_from(u, a, G)
    R = rot_about(-a, theta)
    T = np.eye(4)
    T[:3, :3] = R @ T0[:3, :3]
    T[:3, 3] = pivot + R @ (T0[:3, 3] - pivot)
    return T


def lever_angle_estimate(ctx: Ctx) -> float:
    """Handle angle from where the pinching hand is (rigid hold): atan2 of the grip point about the pivot."""
    m = ctx.mon["tap"]
    r = m.tcp[:3, 3] - ctx.bel.lever_pivot
    d, _ = lever_dirs(ctx.bel.lever_yaw)
    phi = np.arctan2(r[2], float(r @ d))
    phi0 = np.arctan2(LEVER_GRIP_Z, L.LEVER_GRIP_DIST)
    return float(phi - phi0)


def s_lever_ready(ctx: Ctx):
    kin = ctx.kin["tap"]
    P, psi = ctx.bel.lever_pivot, ctx.bel.lever_yaw
    T_g = lever_grip_frame(P, psi)
    u = T_g[:3, 2]
    T_pre = T_g.copy()
    T_pre[:3, 3] = T_g[:3, 3] - u * 0.055 + Z_UP * 0.012
    T_hi = T_pre.copy()
    T_hi[2, 3] += 0.07
    q0 = ctx.q("tap")
    seeds = [np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.5, 1.2, 0.8, 0.6, 0.0, 0.0]), np.array([0.3, 2.0, 1.6, 1.4, 0.2, 0.0]), q0]
    r_pre = kin.ik_multi(T_pre, seeds)
    r_hi = kin.ik_multi(T_hi, seeds + [r_pre.q])
    r_g = kin.ik_multi(T_g, [r_pre.q] + seeds)
    if not (r_pre.ok and r_hi.ok and r_g.ok):
        raise SkillFault(f"handle unreachable for the tap arm at yaw {np.degrees(psi):.0f} deg ({r_pre.reason or r_hi.reason or r_g.reason})")
    traj = M.via_move([q0, r_hi.q, r_pre.q], speed_scale=ctx.speed, min_T=1.0)
    ctx.check_traj("tap", traj, grip_width=ctx.width("tap"))
    yield from ctx.play("tap", traj)
    yield from ctx.grip_open("tap", 0.050)          # open only once the hand is standing clear of the tank
    ctx.scratch["lever_q_pre"] = r_pre.q
    return {"msg": f"fingers open 5 cm beside the handle (yaw {np.degrees(psi):.0f} deg)"}


def s_lever_hold(ctx: Ctx):
    kin = ctx.kin["tap"]
    P, psi = ctx.bel.lever_pivot, ctx.bel.lever_yaw
    T_g = lever_grip_frame(P, psi)
    T0 = ctx.tcp("tap")
    traj = M.cart_move(kin, [T0, T_g], ctx.q("tap"), v_cart=L.APPROACH_SPEED_M_S, speed_scale=ctx.speed, min_T=0.8)
    ctx.monitor("tap", True)
    u = T_g[:3, 2]

    def guard() -> bool:
        return float(-(ctx.mon["tap"].force @ u)) > 5.0

    done = yield from ctx.play("tap", traj, guard=guard)
    if not done:
        ctx.monitor("tap", False)
        raise SkillFault("fingers jammed on the handle while closing in (force above 5 N)")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    res = yield from ctx.grip_close("tap", L.FORCE_CAP_RIM_AND_LEVER_N, rate=0.03)
    w = res["width"]
    if not res["stalled"]:
        ctx.monitor("tap", False)
        raise SkillFault("fingers closed on nothing: the handle is not between the jaws")
    if not (0.008 < w < 0.024):
        ctx.monitor("tap", False)
        raise SkillFault(f"jaws stalled at {fmt_mm(w)}; the handle is 15 mm wide")
    ctx.lever_held = True
    return {"msg": f"fingers closed on the handle, jaws stalled at {fmt_mm(w)} with {res['force']:.1f} N"}


def _lever_arc(ctx: Ctx, th0: float, th1: float, duration_floor: float, label: str, extra_beyond: float = 0.0):
    kin = ctx.kin["tap"]
    P, psi = ctx.bel.lever_pivot, ctx.bel.lever_yaw
    n = max(int(abs(th1 - th0) / 0.12), 2)
    thetas = np.linspace(th0, th1, n + 1)
    poses = [lever_grip_frame(P, psi, th) for th in thetas]
    T_now = ctx.tcp("tap")
    traj = M.cart_move(kin, [T_now] + poses, ctx.q("tap"), v_cart=0.034, speed_scale=ctx.speed, min_T=duration_floor, step_m=0.003)
    ctx.check_traj("tap", traj, grip_width=ctx.width("tap"))
    return traj, thetas


def s_lever_open(ctx: Ctx):
    traj, thetas = _lever_arc(ctx, 0.0, LEVER_OPEN_THETA, 1.6, "open")
    w0 = ctx.width("tap")
    ctx.monitor("tap", True)
    ctx.flow_belief = 0.0
    ctx.flow_clock_on = True
    yield from ctx.play("tap", traj)
    if ctx.width("tap") > w0 + 0.003:
        raise SkillFault("handle slipped out of the fingers while lifting")
    est = lever_angle_estimate(ctx)
    ctx.scratch["lever_theta_open"] = est
    # plan the closing swing now so the pour clock knows how long the flow stays open while it comes down
    close, th = _lever_arc(ctx, est, -0.03, 1.4, "close")
    ctx.scratch["close_traj"] = close
    kin = ctx.kin["tap"]
    lead = close.duration
    for k in range(0, len(close), 4):
        r = kin.fk(close.q[k])[:3, 3] - ctx.bel.lever_pivot
        d, _ = lever_dirs(ctx.bel.lever_yaw)
        phi = np.arctan2(r[2], float(r @ d)) - np.arctan2(LEVER_GRIP_Z, L.LEVER_GRIP_DIST)
        if phi < L.LEVER_FLOW_THRESHOLD:
            lead = k * close.dt
            break
    ctx.scratch["close_lead"] = lead
    return {"msg": f"handle lifted to {np.degrees(est):.0f} deg (flow starts at {np.degrees(L.LEVER_FLOW_THRESHOLD):.0f} deg); closing swing keeps it open {lead:.1f} s"}


def s_pour_wait(ctx: Ctx):
    lead = float(ctx.scratch.get("close_lead", 1.0))
    t0 = ctx.plant.time
    while ctx.flow_belief < ctx.pour_s - lead:
        if ctx.plant.time - t0 > ctx.pour_s * 2 + 6:
            raise SkillFault("pour clock never started: handle angle estimate stayed below the flow angle")
        ctx.hold("tap")
        yield
    return {"msg": f"flow clock {ctx.flow_belief:.2f} s, {lead:.1f} s more while the handle swings down"}


def s_lever_close(ctx: Ctx):
    traj = ctx.scratch.get("close_traj")
    if traj is None:
        th = ctx.scratch.get("lever_theta_open", LEVER_OPEN_THETA)
        traj, _ = _lever_arc(ctx, th, -0.03, 1.4, "close")
    # restart from where the hand actually is: prepend a tiny correction so there is no jump
    ctx.check_traj("tap", traj, grip_width=ctx.width("tap"))
    yield from ctx.play("tap", traj)
    yield from ctx.wait(0.2)
    ctx.flow_clock_on = False
    est = lever_angle_estimate(ctx)
    return {"msg": f"handle back down against its stop (estimated {np.degrees(est):.0f} deg), flow clock {ctx.flow_belief:.2f} s"}


def s_lever_release(ctx: Ctx):
    yield from ctx.grip_open("tap", 0.050)
    ctx.lever_held = False
    kin = ctx.kin["tap"]
    T0 = ctx.tcp("tap")
    u = T0[:3, 2]
    T1 = T0.copy()
    T1[:3, 3] = T0[:3, 3] - u * 0.06 + Z_UP * 0.012
    T2 = T1.copy()
    T2[2, 3] += 0.06
    traj = M.cart_move(kin, [T0, T1, T2], ctx.q("tap"), v_cart=0.07, speed_scale=ctx.speed, min_T=0.8)
    ctx.check_traj("tap", traj)
    yield from ctx.play("tap", traj)
    ctx.monitor("tap", False)
    return {"msg": "fingers open, hand lifted off the handle"}
