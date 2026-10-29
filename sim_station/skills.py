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
LEVER_TILT_DEG = 50.0                          # steeper pinch: the 40 deg arc misses the last of the lift from the closer base
LEVER_REV_TILT_DEG = 10.0                      # reverse pinch: ends the 60 deg lift at 70 deg, inside the wrist's reach
LEVER_OPEN_THETA = 1.05                        # how far the handle is lifted (flow starts at 0.45)
LEVER_GRIP_Z = 0.0015 + 0.5 * L.LEVER_HANDLE_THICK
LEVER_SEEDS = [np.array(x, float) for x in ([0.6, 1.8, 1.3, 0.9, 0, 0], [0.5, 1.2, 0.8, 0.6, 0, 0], [0.3, 2.0, 1.6, 1.4, 0.2, 0],
                                            [0.0, 1.5, 1.2, 0.5, 0, 0], [-0.5, 1.5, 1.2, 0.5, 1.0, 0],
                                            [1.0, 1.5, 1.2, 0.5, -1.0, 0], [0.2, 1.0, 0.6, 1.2, 0, 0])]


def lever_frame(ctx: Ctx, psi: float, theta: float = 0.0) -> np.ndarray:
    """Grip frame in whichever pinch (forward or reverse) the search closed on the handle with."""
    return lever_grip_frame(np.asarray(ctx.bel.lever_pivot, float), psi, theta,
                            reverse=bool(ctx.scratch.get("lever_rev", False)))


def side_frame(kin, xy, z, off=-9.0, tilt=0.0, depth=0.0) -> np.ndarray:
    base = kin.mount[:2]
    az = np.arctan2(xy[1] - base[1], xy[0] - base[0]) + np.radians(off)
    t = np.radians(tilt)
    a = np.array([np.cos(az) * np.cos(t), np.sin(az) * np.cos(t), np.sin(t)])
    j = np.array([-np.sin(az), np.cos(az), 0.0])
    return frame_from(a, j, np.array([xy[0], xy[1], z]) + a * depth)


def pour_frame(ctx: Ctx, kin, xy, z, tilt=0.0, depth=0.0) -> np.ndarray:
    """The spout and coaster frames. `tilt` is the cup's lean; `hold_pitch` is the hand pitch that holds this grasp's cup vertical."""
    T = side_frame(kin, xy, float(z), tilt=tilt + float(ctx.scratch.get("hold_pitch") or 0.0), depth=depth)
    off = ctx.scratch.get("hold_offset")
    if off is not None and ctx.bel.cup_in_hand:
        # Put the cup, not the fingertip point, on the target.
        T[:3, 3] = T[:3, 3] - T[:3, :3] @ np.asarray(off, float)
    return T


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


def s_rest_over_cup(ctx: Ctx):
    """Fold the cup arm after serving. When the straight fold would swing the housing through the served cup,
    rise well above the rim first."""
    try:
        return (yield from s_goto_rest(ctx, "cup"))
    except PlanError:
        pass
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    for z in (0.24, 0.20, 0.28):
        T1 = T0.copy()
        T1[2, 3] = max(float(T0[2, 3]), z)
        try:
            up = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.5, max_joint_step=0.25)
            ctx.check_traj("cup", up)
        except PlanError:
            continue
        yield from ctx.play("cup", up)
        break
    return (yield from s_goto_rest(ctx, "cup"))


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
    if ctx.scratch.get("wide_search") and az is None:
        # A loose cup can be taken from any side. A level hand pinches it without tipping it, so a flatter approach from
        # further round wins over a steep one straight from the base. The slide in must stay on one IK branch.
        fr, last = None, ""
        for tilt in (0.0, -8.0, -15.0, -22.0, -30.0, -45.0):
            for off in (-9.0, 0.0, -20.0, -35.0, 20.0, 35.0, -50.0, 50.0):
                try:
                    cand = ctx.side_frames("cup", xy, z, depth=0.004, standoff=0.05, tilts=(tilt,), offsets=(off,))
                    M.cart_move(ctx.kin["cup"], [cand["pre"], cand["grasp"]], cand["q"]["pre"], v_cart=0.015,
                                speed_scale=ctx.speed)
                except PlanError as exc:
                    last = str(exc)
                    continue
                fr = cand
                break
            if fr is not None:
                break
        if fr is None:
            raise SkillFault(f"no side approach to the cup slides in on one arm branch ({last})")
    else:
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


LOOSE_PUSH_STEP_N = 1.5


def s_slide_in(ctx: Ctx):
    """Slide the open fingers in around the cup wall. If a fingertip lands on the rim, comply sideways (admittance on the
    lateral contact force) so the lead-in funnel can centre the hand on the cup instead of pushing the cup away.

    A loose cup (Level 5) is not held by the dish. A few newtons, which a coaster shrug off, slide it across the mat,
    so that slide stops at the first push and backs off for another try.
    """
    fr = ctx.scratch["cup_fr"]
    kin = ctx.kin["cup"]
    q0 = ctx.q("cup")
    loose = bool(ctx.scratch.get("wide_search"))
    v = 0.015 if loose else L.APPROACH_SPEED_M_S
    push_cap = 5.0
    try:
        traj = M.cart_move(kin, [ctx.tcp("cup"), fr["grasp"]], q0, v_cart=v, speed_scale=ctx.speed)
    except PlanError:
        # The settled hand sits a hair off the planned standoff; the plan from the standoff itself is on one branch.
        traj = M.cart_move(kin, [fr["pre"], fr["grasp"]], fr["q"]["pre"], v_cart=v, speed_scale=ctx.speed)
    ctx.monitor("cup", True)
    a, j = fr["a"], fr["j"]
    j_base = kin.R_wb.T @ j
    delta = 0.0
    dt = CTRL_STEPS * ctx.plant.dt
    jammed = False
    bumped = False
    pushed = 0
    for k_ in range(len(traj)):
        m = ctx.mon["cup"]
        if loose:
            # The residual carries a model offset of about a newton while the arm moves. A wall met by a pad is a step.
            pushed = pushed + 1 if (k_ * dt > 0.15 and float(-(m.delta_force(25) @ a)) > LOOSE_PUSH_STEP_N) else 0
            if pushed >= 3:
                bumped = True
                break
        elif float(-(m.force @ a)) > push_cap:
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
    if bumped:
        T0 = ctx.tcp("cup")
        T_back = T0.copy()
        T_back[:3, 3] = T0[:3, 3] - a * 0.040
        try:
            back = M.cart_move(kin, [T0, T_back], ctx.q("cup"), v_cart=0.04, speed_scale=ctx.speed, min_T=0.35)
            ctx.check_traj("cup", back, grip_width=ctx.width("cup"))
            yield from ctx.play("cup", back)
        except PlanError:
            pass
        ctx.event("loose cup started to slide; backed the hand off to take another line")
        return {"msg": "met the cup early and backed off so it would not be pushed across the mat"}
    if delta != 0.0:
        fr["grasp"] = fr["grasp"].copy()
        fr["grasp"][:3, 3] = fr["grasp"][:3, 3] + j * delta
        ctx.event(f"fingertip met the rim: complied {1000 * delta:+.0f} mm sideways to centre on the cup")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    return {"msg": "fingers straddle the cup wall" + (f" (centred {1000 * delta:+.0f} mm by touch)" if delta != 0.0 else "")}


CHORD_ACCEPT_S = 0.008          # a 66 mm stall is 7 mm off the axis of the 67.4 mm cup; shuffling the hand after that knocks the cup away


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
    tries = 9
    if ctx.scratch.get("wide_search"):
        fallback = fallback + [(0.024, 0.0), (-0.024, 0.0), (0.0, 0.024), (0.0, -0.024), (0.036, 0.0), (-0.036, 0.0)]
        tries = 12
    lat, deep = 0.0, 0.0
    sign, s_prev, n = 1.0, None, 0
    last = ""
    fb = 0
    while n < tries:
        res = yield from ctx.grip_close("cup", cap, rate=0.05)
        w = res["width"]
        s = float(np.sqrt(max(r_c * r_c - (0.5 * w) ** 2, 0.0))) if res["stalled"] else None
        if res["stalled"] and 0.050 < w < 0.075 and s <= CHORD_ACCEPT_S:
            ctx.bel.cup_in_hand = True
            ctx.scratch["hold_width"] = w
            if ctx.scratch.get("wide_search"):
                # The cup stood vertical when the jaws closed. Carry and pour at the same hand pitch so it stays vertical.
                ctx.scratch["hold_pitch"] = float(fr["tilt"])
                # Where the cup sits in the hand, from the stomped centre and where the hand closed. The jaws centre it
                # across; along the fingers a cup seated in the lead-in still stalls near full width, so the stall
                # width alone cannot say how far back it is.
                T_now = ctx.tcp("cup")
                centre = fr["grasp"][:3, 3] - a * 0.004
                rel = T_now[:3, :3].T @ (centre - T_now[:3, 3])
                rel[1] = 0.0
                ctx.scratch["hold_offset"] = np.clip(rel, -0.025, 0.025)
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
                back = M.cart_move(kin, [T0, T_back], ctx.q("cup"), v_cart=0.06, speed_scale=ctx.speed, min_T=0.7)
                ctx.check_traj("cup", back, grip_width=L.GRIPPER_MAX_OPEN_M, stride_s=0.01)
                yield from ctx.play("cup", back)
            except PlanError:
                pass
        T_g = fr["grasp"].copy()
        T_g[:3, 3] = T_g[:3, 3] + j * lat + a * deep
        try:
            move = M.cart_move(kin, [ctx.tcp("cup"), T_g], ctx.q("cup"), v_cart=0.04 if ctx.scratch.get("wide_search") else 0.05,
                               speed_scale=ctx.speed, min_T=0.7)
            ctx.check_traj("cup", move, grip_width=L.GRIPPER_MAX_OPEN_M, stride_s=0.01)
            if ctx.scratch.get("wide_search"):
                ctx.monitor("cup", True)
                state = {"t": 0}

                def on_tick(k, state=state) -> None:
                    state["t"] = k

                def guard(state=state) -> bool:
                    if state["t"] * move.dt < 0.15:
                        return False
                    state["n"] = state.get("n", 0) + 1 if float(-(ctx.mon["cup"].delta_force(25) @ a)) > LOOSE_PUSH_STEP_N else 0
                    return state["n"] >= 3

                done = yield from ctx.play("cup", move, guard=guard, on_tick=on_tick)
                ctx.monitor("cup", False)
                if not done:
                    ctx.event("stopped short so the loose cup would not be pushed")
            else:
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
        for tilt in (0.0, -8.0, 8.0, -12.0):
            try:
                ok = True
                for z in (0.118, spout_z() + 0.002, spout_z()):
                    if not kin.ik_multi(pour_frame(ctx, kin, e, z, tilt=tilt), seeds).ok:
                        ok = False
                        last = f"entry ({e[0]:.3f}, {e[1]:.3f}) tilt {tilt:.0f}: unreachable"
                        break
                if not ok:
                    continue
                p1 = pour_frame(ctx, kin, e, 0.118, tilt=tilt)
                p2 = pour_frame(ctx, kin, e, spout_z() + 0.002, tilt=tilt)
                carry = M.cart_move(kin, [T0, p1, p2], q0, v_cart=L.CARRY_SPEED_M_S * 1.3, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.25)
                ctx.check_traj("cup", carry)
                # The slide that actually runs dips 4 mm first. A corridor whose wrist is already
                # on the stop can draw the straight line and still be unable to dip.
                p_under = pour_frame(ctx, kin, np.array([xs, ys]), spout_z(), tilt=tilt)
                p_dip = p2.copy()
                p_dip[2, 3] = float(p_under[2, 3]) - 0.004
                p_mid = p_under.copy()
                p_mid[2, 3] = float(p_under[2, 3]) - 0.004
                slide = M.cart_move(kin, [p2, p_dip, p_mid, p_under], carry.q[-1], v_cart=0.08,
                                    speed_scale=ctx.speed, min_T=1.2, max_joint_step=0.2)
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
    p1 = pour_frame(ctx, kin, e, z_carry, tilt=tilt)
    p2 = pour_frame(ctx, kin, e, spout_z() + 0.002, tilt=tilt)
    traj = M.cart_move(kin, [T0, p1, p2], ctx.q("cup"), v_cart=L.CARRY_SPEED_M_S * 1.3, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.25)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    return {"msg": f"cup lowered to spout height in the corridor ({ctx.scratch['spout_side']} side of the dispenser)"}


def s_slide_under(ctx: Ctx):
    kin = ctx.kin["cup"]
    xs, ys = ctx.bel.spout_xy
    T0 = ctx.tcp("cup")
    tilt = ctx.scratch.get("spout_tilt", 0.0)
    # The carry can leave the cup leaning a few degrees in the fingers. The high side of the rim
    # then meets the spout tip if the slide stays at the final height, so come in low and rise
    # only once the tip is over the mouth. A grasp that sits a millimetre outside the wrist's
    # reach at that height gets the next tries, a couple of millimetres higher.
    traj = None
    last = "no slide"
    # The first try is the 4 mm dip the nominal pour uses. A grasp that leaves the wrist
    # on its stop cannot dip, so the later tries come in level, a little higher.
    for lift, step, dip, creep in ((0.0, 0.2, 0.004, False), (0.003, 0.24, 0.004, False), (0.006, 0.24, 0.004, False),
                                   (0.0, 0.25, 0.006, True), (0.004, 0.28, 0.004, True)):
        T1 = pour_frame(ctx, kin, np.array([xs, ys]), spout_z(), tilt=tilt)
        T1[2, 3] = float(T1[2, 3]) + lift
        z_low = float(T1[2, 3]) - dip
        T_dip = T0.copy()
        T_dip[2, 3] = z_low
        T_mid = T1.copy()
        T_mid[2, 3] = z_low
        if creep:
            # The wrist is on its stop and cannot dip in place. Move toward the outlet first,
            # then drop, so the rim still passes under the tip.
            aim = T1[:2, 3] - T0[:2, 3]
            n = float(np.linalg.norm(aim))
            step_xy = aim / n * min(0.03, 0.4 * n) if n > 1e-6 else np.zeros(2)
            Ta = T0.copy()
            Ta[0, 3] = float(T0[0, 3] + step_xy[0])
            Ta[1, 3] = float(T0[1, 3] + step_xy[1])
            Td = Ta.copy()
            Td[2, 3] = z_low
            poses = [T0, Ta, Td, T_mid, T1]
        else:
            poses = [T0, T_dip, T_mid, T1]
        try:
            traj = M.cart_move(kin, poses, ctx.q("cup"), v_cart=0.08, speed_scale=ctx.speed, min_T=1.2, max_joint_step=step)
            ctx.check_traj("cup", traj)
            break
        except PlanError as exc:
            last = str(exc)
            traj = None
            ctx.event(f"slide try at {lift * 1000:.0f} mm higher, dip {dip * 1000:.0f} mm"
                      f"{', after a short creep' if creep else ''}: {exc}")
    if traj is None:
        raise SkillFault(f"the slide under the spout is not reachable ({last})")
    yield from ctx.play("cup", traj)
    yield from ctx.wait(0.25)
    ctx.scratch["under_spout_tcp"] = ctx.tcp("cup")[:3, 3].copy()
    return {"msg": "cup centred under the outlet, rim 1.5 cm below it"}


def s_exit_spout(ctx: Ctx):
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    e = ctx.scratch["spout_entry"]
    tilt = ctx.scratch.get("spout_tilt", 0.0)
    p1 = pour_frame(ctx, kin, e, spout_z() + 0.002, tilt=tilt)
    z_low = float(p1[2, 3]) - 0.004
    T_dip = T0.copy()
    T_dip[2, 3] = z_low
    p_low = pour_frame(ctx, kin, e, spout_z() + 0.002, tilt=tilt)
    p_low[2, 3] = z_low
    p2 = pour_frame(ctx, kin, e, 0.118, tilt=tilt)
    traj = M.cart_move(kin, [T0, T_dip, p_low, p1, p2], ctx.q("cup"), v_cart=0.08, speed_scale=ctx.speed, min_T=1.2, max_joint_step=0.2)
    ctx.check_traj("cup", traj)
    yield from ctx.play("cup", traj)
    return {"msg": "slid out sideways under the outlet, then lifted clear"}


def s_to_coaster(ctx: Ctx, seat_z: float):
    kin = ctx.kin["cup"]
    cx, cy = ctx.bel.coaster_xy
    T0 = ctx.tcp("cup")
    seeds = [ctx.q("cup"), L.REAL_L1_CUP_WAYPOINTS["in_coaster"][:6], np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0])]
    chosen = None
    for tilt in (0.0, -8.0, -15.0):
        top = pour_frame(ctx, kin, np.array([cx, cy]), seat_z + HOLD_DZ + 0.035, tilt=tilt)
        low = pour_frame(ctx, kin, np.array([cx, cy]), seat_z + HOLD_DZ - 0.006, tilt=tilt)
        if kin.ik_multi(top, seeds).ok and kin.ik_multi(low, seeds).ok:
            chosen = (tilt, top)
            break
    if chosen is None:
        raise SkillFault("the coaster cannot be reached with the cup upright")
    ctx.scratch["coaster_tilt"] = chosen[0]
    z_mid = max(top_over(ctx) + 0.08, 0.28) if ctx.bel.stack_n is not None else 0.22
    z_over = z_mid
    up = T0.copy()
    up[2, 3] = max(float(T0[2, 3]), z_over)
    across = chosen[1].copy()
    across[2, 3] = z_over
    # a waypoint out beside the dispenser, so the carry does not cut across the stack on the coaster spot
    side = pour_frame(ctx, kin, np.array([min(cx, L.MOUTH_WORLD_XY[0]) - 0.06, L.MOUTH_WORLD_XY[1]]), z_mid, tilt=chosen[0])
    routes = [
        [T0, up, across, chosen[1]],
        [T0, up, side, across, chosen[1]],
        [T0, chosen[1]],
    ]
    last = ""
    traj = None
    for poses in routes:
        try:
            traj = M.cart_move(kin, poses, ctx.q("cup"), v_cart=L.CARRY_SPEED_M_S * 1.4, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.22)
            ctx.check_traj("cup", traj)
            break
        except PlanError as exc:
            last = str(exc)
            traj = None
    if traj is None:
        raise SkillFault(f"no way to the coaster that stays off the stack ({last})")
    yield from ctx.play("cup", traj)
    return {"msg": "cup held 3.5 cm above the coaster"}


def top_over(ctx: Ctx) -> float:
    """Height of the top of whatever is left of the stack, from the stack count the plan believes in."""
    n = max((ctx.bel.stack_n or 1) - 1, 1)
    return L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP + L.CUP_HEIGHT


def s_place_cup(ctx: Ctx, seat_z: float):
    """Guarded descent onto the bolted dish. The coaster does not move, so the hand goes straight down."""
    kin = ctx.kin["cup"]
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
        if z_now > expected + 0.016:                   # armed only within 16 mm of the believed seat
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
    # The dish is bolted to the mat. It does not move, so there is nothing to feel for and no sideways
    # correction. The cup is let go on the axis it was carried in on.
    ctx.monitor("cup", False)
    z_end = float(ctx.tcp("cup")[2, 3])
    expected = seat_z + HOLD_DZ
    if not done:
        return {"msg": f"cup base touched down at TCP z {z_end * 1000:.1f} mm (expected {expected * 1000:.1f} mm)"}
    return {"msg": f"cup lowered to the seat without a contact spike, TCP z {z_end * 1000:.1f} mm"}


def belief_walk_ok(shift_xy, err_xy) -> bool:
    """True when a correction toward the believed coaster is not fighting the dish push."""
    err = np.asarray(err_xy, dtype=float)
    if float(np.linalg.norm(err)) <= 0.003:
        return False
    return float(np.dot(np.asarray(shift_xy, dtype=float), err)) >= -1e-9


def s_release(ctx: Ctx):
    w = yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M, cap=L.FORCE_CAP_PROBE_N)
    ctx.bel.cup_in_hand = False
    return {"msg": f"jaws opened to {fmt_mm(float(w))}, cup released"}


def s_retreat(ctx: Ctx, options=((0.0, 0.07), (0.04, 0.08), (0.06, 0.09), (0.08, 0.07), (0.0, 0.03))):
    """Lift clear of the rim first. The dish is fixed, and a sideways drag at seat height walks the cup off centre."""
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    a_h = T0[:3, 2].copy()
    a_h[2] = 0.0
    a_h = a_h / max(np.linalg.norm(a_h), 1e-6)
    last = ""
    # A pure lift, then the older back-then-up escapes if the lift is blocked.
    T_up = T0.copy()
    T_up[2, 3] = float(T0[2, 3]) + 0.028
    T_back = T_up.copy()
    T_back[:3, 3] = T_up[:3, 3] - a_h * 0.045
    T_clear = T_back.copy()
    T_clear[2, 3] = float(T_back[2, 3]) + 0.07
    try:
        lift = M.cart_move(kin, [T0, T_up, T_back, T_clear], ctx.q("cup"), v_cart=0.06, speed_scale=ctx.speed,
                           min_T=0.7, max_joint_step=0.18, step_m=0.006)
        ctx.check_traj("cup", lift)
        yield from ctx.play("cup", lift)
        return {"msg": "hand lifted clear of the rim, then backed away"}
    except PlanError as exc:
        last = str(exc)
    for back, up in options:
        pts = [T0]
        if back > 1e-4:
            Tb = T0.copy()
            Tb[:3, 3] = T0[:3, 3] - a_h * back
            pts.append(Tb)
        T1 = pts[-1].copy()
        T1[:3, 3] = pts[-1][:3, 3] + Z_UP * up
        pts.append(T1)
        try:
            traj = M.cart_move(kin, pts, ctx.q("cup"), v_cart=0.06, speed_scale=ctx.speed, min_T=0.7,
                               max_joint_step=0.18, step_m=0.006)
            ctx.check_traj("cup", traj)
        except PlanError as exc:
            last = str(exc)
            continue
        yield from ctx.play("cup", traj)
        return {"msg": f"hand backed out {back * 100:.0f} cm, then raised {up * 100:.0f} cm clear of the served cup"}
    # A pitched hold can leave the wrist on its stop for a straight line. Short joint-space steps up and back
    # stay within a few millimetres of that line and do not need the wrist to pass its stop.
    seeds = [ctx.q("cup"), L.REAL_L1_CUP_WAYPOINTS["above_coaster"][:6], L.REAL_L1_CUP_WAYPOINTS["in_coaster"][:6]]
    for back, up in ((0.0, 0.04), (0.02, 0.05), (0.05, 0.07), (0.0, 0.08)):
        T1 = T0.copy()
        T1[:3, 3] = T0[:3, 3] - a_h * back + Z_UP * up
        r = kin.ik_multi(T1, seeds)
        if not r.ok:
            last = r.reason or last
            continue
        try:
            traj = M.joint_move(ctx.q("cup"), r.q, speed_scale=ctx.speed, min_T=1.0)
            ctx.check_traj("cup", traj)
        except PlanError as exc:
            last = str(exc)
            continue
        yield from ctx.play("cup", traj)
        return {"msg": f"hand raised {up * 100:.0f} cm clear of the served cup in joint space"}
    raise SkillFault(f"no reachable way to pull the hand out of the cup: {last}")


# ---------------------------------------------------------------------------
# tap arm: pinch the handle, lift it, hold, close it, let go
# ---------------------------------------------------------------------------

def lever_dirs(psi: float):
    d = np.array([np.cos(psi), np.sin(psi), 0.0])
    a = np.array([-np.sin(psi), np.cos(psi), 0.0])
    return d, a


def lever_grip_frame(pivot: np.ndarray, psi: float, theta: float = 0.0, tilt_deg: Optional[float] = None,
                     reverse: bool = False) -> np.ndarray:
    """Pinch on the paddle. Forward: the hand reaches past the knob and points down the handle.
    Reverse: the hand comes from beyond the tip and points back at the knob, so a handle facing the
    tap arm can be held. The lift rotates the hand about the jaw axis, which adds the lift angle to
    the reverse tilt; that is why the reverse pinch starts nearly level."""
    d, a = lever_dirs(psi)
    if tilt_deg is None:
        tilt_deg = LEVER_REV_TILT_DEG if reverse else LEVER_TILT_DEG
    t = np.radians(tilt_deg)
    if reverse:
        u, jaw = -d * np.cos(t) - Z_UP * np.sin(t), -a
    else:
        u, jaw = d * np.cos(t) - Z_UP * np.sin(t), a
    G = pivot + d * L.LEVER_GRIP_DIST + Z_UP * LEVER_GRIP_Z
    T0 = frame_from(u, jaw, G)
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
    T_g = lever_frame(ctx, psi)
    u = T_g[:3, 2]
    q0 = ctx.q("tap")
    seeds = LEVER_SEEDS[:3] + [q0] + LEVER_SEEDS[3:]
    r_g = kin.ik_multi(T_g, seeds)
    chosen = None
    # The first candidate is the standoff the pour has always used. Shorter ones are only for a handle
    # that points where that hover cannot be reached.
    for back, lift, z_up in ((0.055, 0.07, 0.012), (0.040, 0.05, 0.010), (0.028, 0.0, 0.010)):
        T_pre = T_g.copy()
        T_pre[:3, 3] = T_g[:3, 3] - u * back + Z_UP * z_up
        r_pre = kin.ik_multi(T_pre, ([r_g.q] if r_g.ok else []) + seeds)
        if not (r_pre.ok and r_g.ok):
            continue
        hops = [r_pre.q]
        if lift > 0:
            T_hi = T_pre.copy()
            T_hi[2, 3] = float(T_pre[2, 3]) + lift
            r_hi = kin.ik_multi(T_hi, [r_pre.q] + seeds)
            if r_hi.ok:
                hops = [r_hi.q, r_pre.q]
            elif back == 0.055:
                continue
        chosen = (T_pre, r_pre, hops)
        break
    if chosen is None:
        raise SkillFault(f"handle unreachable for the tap arm at yaw {np.degrees(psi):.0f} deg ({r_g.reason or 'no standoff'})")
    T_pre, r_pre, hops = chosen
    traj = None
    for use in (hops, hops[-1:]):
        try:
            traj = M.via_move([q0] + use, speed_scale=ctx.speed, min_T=0.8)
            ctx.check_traj("tap", traj, grip_width=ctx.width("tap"))
            break
        except PlanError:
            traj = None
    if traj is None:
        raise SkillFault(f"no clear path to the handle at yaw {np.degrees(psi):.0f} deg")
    yield from ctx.play("tap", traj)
    # 32 mm clears the 15 mm paddle. 50 mm puts a fingertip on the tank once the base is a few millimetres off.
    yield from ctx.grip_open("tap", 0.032, cap=8.0)
    ctx.scratch["lever_q_pre"] = r_pre.q
    return {"msg": f"fingers open 3.2 cm beside the handle (yaw {np.degrees(psi):.0f} deg)"}


def s_lever_hold(ctx: Ctx):
    kin = ctx.kin["tap"]
    P, psi = ctx.bel.lever_pivot, ctx.bel.lever_yaw
    T_g = lever_frame(ctx, psi)
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
    # The jaws overshoot and the paddle springs them back. The close can report "not stalled" while the
    # settled gap is the paddle. Trust that gap. Walking around to regrasp is what spins the handle off it.
    yield from ctx.wait(0.3)
    w = float(ctx.width("tap"))
    ctx.event(f"handle close settled at {fmt_mm(w)}")
    if not (0.0120 < w < 0.0205):
        w = yield from _regrasp_paddle(ctx, w)
    ctx.lever_held = True
    return {"msg": f"fingers closed on the handle, jaws stalled at {fmt_mm(w)} with {res['force']:.1f} N"}


def _regrasp_paddle(ctx: Ctx, width: float) -> float:
    """The close missed the 15 mm paddle. Step around the hinge and close again. Do not lift until the width is right."""
    kin = ctx.kin["tap"]
    psi0 = float(ctx.bel.lever_yaw)
    last = fmt_mm(width)
    # A few millimetres and a few degrees. A wide walk turns the handle and then closes on nothing.
    for dy in (0.0, 0.006, -0.006):
        for deg in (0.0, 6.0, -6.0):
            if dy == 0.0 and deg == 0.0:
                continue
            psi = float(_wrap_pi(psi0 + np.radians(deg)))
            T = lever_frame(ctx, psi)
            T[:3, 3] = T[:3, 3] + T[:3, 1] * dy
            yield from ctx.grip_open("tap", 0.028, cap=6.0)
            try:
                traj = M.cart_move(kin, [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.03, speed_scale=ctx.speed,
                                   min_T=0.35, max_joint_step=0.16)
                ctx.check_traj("tap", traj, grip_width=0.028)
            except PlanError:
                continue
            yield from ctx.play("tap", traj)
            res = yield from ctx.grip_close("tap", L.FORCE_CAP_RIM_AND_LEVER_N, rate=0.03)
            yield from ctx.wait(0.25)
            w = float(ctx.width("tap"))
            last = fmt_mm(w)
            if w < 0.004:
                # The paddle is no longer between the fingers. Stop before the walk shoves the valve.
                ctx.monitor("tap", False)
                raise SkillFault(f"jaws stalled at {last}; the handle is 15 mm wide")
            if 0.0120 < w < 0.0205:
                ctx.bel.lever_yaw = psi
                ctx.event(f"re-closed on the paddle at {np.degrees(psi):.0f} deg, jaws {last}")
                return float(w)
    ctx.monitor("tap", False)
    raise SkillFault(f"jaws stalled at {last}; the handle is 15 mm wide")


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _lever_arc(ctx: Ctx, th0: float, th1: float, duration_floor: float, label: str, extra_beyond: float = 0.0):
    kin = ctx.kin["tap"]
    P, psi = ctx.bel.lever_pivot, ctx.bel.lever_yaw
    n = max(int(abs(th1 - th0) / 0.12), 2)
    thetas = np.linspace(th0, th1, n + 1)
    poses = [lever_frame(ctx, psi, th) for th in thetas]
    T_now = ctx.tcp("tap")
    # 0.20 rad, not the 0.12 default: the first sample of this arc can sit 0.13 rad from the seed on a
    # legal wrist branch. The collision check still rejects a path that would hit the tank.
    traj = M.cart_move(kin, [T_now] + poses, ctx.q("tap"), v_cart=0.034, speed_scale=ctx.speed, min_T=duration_floor,
                       step_m=0.003, max_joint_step=0.20)
    ctx.check_traj("tap", traj, grip_width=ctx.width("tap"))
    return traj, thetas


def _seat_handle(ctx: Ctx):
    """Square the fingers on the paddle before the lift.

    The search reports the radius it closed on, which can be a few degrees off the handle. Lifting on
    that heading raises the hand beside the hinge: the pour clock still runs, and the valve stays shut.
    """
    kin = ctx.kin["tap"]
    P = np.asarray(ctx.bel.lever_pivot, float)
    psi = float(ctx.bel.lever_yaw)
    w0 = float(ctx.width("tap"))
    ctx.monitor("tap", True)
    yield from ctx.wait(0.2)

    def twist() -> float:
        r = ctx.tcp("tap")[:3, 3] - P
        r[2] = 0.0
        nrm = float(np.linalg.norm(r))
        if nrm < 1e-4:
            return 0.0
        tang = np.array([-r[1], r[0], 0.0]) / nrm
        return float(ctx.mon["tap"].force @ tang)

    def nudge(yaw: float) -> bool:
        T = lever_frame(ctx, yaw)
        try:
            traj = M.cart_move(kin, [ctx.tcp("tap"), T], ctx.q("tap"), v_cart=0.02, speed_scale=ctx.speed,
                               min_T=0.28, max_joint_step=0.16)
            ctx.check_traj("tap", traj, grip_width=w0)
        except PlanError:
            return False
        yield from ctx.play("tap", traj)
        yield from ctx.wait(0.12)
        return ctx.width("tap") <= w0 + 0.004

    f0 = twist()
    if abs(f0) < 1.2:
        return
    direction = 0.0
    for sgn in (1.0, -1.0):
        trial = float((psi + sgn * np.radians(3.5) + np.pi) % (2.0 * np.pi) - np.pi)
        held = yield from nudge(trial)
        if not held:
            continue
        f1 = twist()
        if abs(f1) < abs(f0) - 0.25:
            direction = sgn
            psi = trial
            f0 = f1
            break
        yield from nudge(psi)
    if direction == 0.0:
        return
    for _ in range(5):
        if abs(f0) < 1.0:
            break
        nxt = float((psi + direction * np.radians(3.5) + np.pi) % (2.0 * np.pi) - np.pi)
        held = yield from nudge(nxt)
        if not held:
            break
        f1 = twist()
        if abs(f1) > abs(f0) + 0.15:
            yield from nudge(psi)
            break
        psi, f0 = nxt, f1
    ctx.bel.lever_yaw = psi
    ctx.event(f"seated the fingers on the handle at {np.degrees(psi):.0f} deg")


def s_lever_open(ctx: Ctx):
    # Levels that had to find the handle carry the radius they closed on, not a measured yaw.
    if ctx.bel.found.get("lever_yaw_before") is not None:
        yield from _seat_handle(ctx)
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
    # Just clear of the 15 mm paddle. Opening to 5 cm while the hand is still on the valve puts a pad on the tank.
    yield from ctx.grip_open("tap", 0.028, cap=6.0)
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
