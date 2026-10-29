"""Level 2 and 3 skills: unstack the top cup of an inverted nested stack with both arms, lay it on its side, tip it
upright by pivoting about its base edge, and re-grasp it.

Everything here uses sensing only: the cup arm's grasp tells the stack position, the pin pads centre the stack,
the force drop tells when the nested cups let go, and the tap arm's pinch pose tells where the cup lies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.kin import frame_from, rot_about
from sim_station.motion import PlanError
from sim_station.runtime import CTRL_STEPS, Ctx, SkillFault
from sim_station.sensing import Baseline
from sim_station.skills import HOLD_DZ, Z_UP, fmt_mm, side_frame

TOP_GRASP_DEPTH = 0.008           # TCP sits this far below the top cup's base face
PIN_BAND_BELOW_RIM = 0.0135       # tap pad centre below the top cup's rim (the funnel pad is 19 mm tall)
ROLL_SIGN = -1.0
PLACE_TCP_Z = 0.060               # the vertical-jaw housing plate reaches 51 mm below the TCP
FLIP_RELEASE_DEG = 78.0
G = 9.81


def top_face_z(n: int) -> float:
    return L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP + L.CUP_HEIGHT


def stack_rim_z(n: int) -> float:
    return L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP


def hold_frame_at(T: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = T.copy()
    out[:3, 3] = pos
    return out


def roll_frame(T: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a TCP frame about its own approach axis through the TCP."""
    a = T[:3, 2]
    R = rot_about(a, angle)
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = T[:3, 3]
    return out


# ---------------------------------------------------------------------------
# cup arm: take hold of the top cup
# ---------------------------------------------------------------------------

# Negative azimuth first: the wrist is turned the other way before the fingers close.
# tilt -8 deg with a 65 deg offset is the attitude whose rising roll ends within a degree of vertical.
PREGRASP = ((-65, -8), (65, -8), (-70, -8), (70, -8), (-65, -12), (65, -12), (-60, -20), (60, -20))


def _flip_seeds(sgn: float):
    return [
        np.array([0.0, 1.2, 1.1, 0.5, 0.0, -1.6 * sgn]),
        np.array([0.3, 1.0, 1.0, 0.8, 0.6, -1.6 * sgn]),
        np.array([0.1, 1.6, 1.2, 0.8, 0.0, -1.4 * sgn]),
        np.array([0.4, 1.8, 1.3, 0.6, -0.4, -1.5 * sgn]),
        np.array([0.2, 1.4, 1.0, 0.4, 0.3, -1.2 * sgn]),
    ]


def upright_frame(kin, xy, z, off_deg: float = 0.0) -> np.ndarray:
    """Level side grasp rolled half a turn, so the hand's x axis points up and the cup stands flat."""
    T = side_frame(kin, np.asarray(xy, float), z, off=off_deg, tilt=0.0)
    out = np.eye(4)
    out[:3, :3] = rot_about(T[:3, 2], np.pi) @ T[:3, :3]
    out[:3, 3] = T[:3, 3]
    return out


def _chain(kin, poses, seeds, q0, jump: float = 0.55):
    qprev = np.asarray(q0, float)
    qs = []
    for T in poses:
        r = kin.ik_multi(T, [qprev] + list(seeds))
        if not r.ok:
            return None, r.reason
        if qs and float(np.max(np.abs(r.q - qs[-1]))) > jump:
            return None, "joint jump"
        qs.append(r.q)
        qprev = r.q
    return qs, ""


def flip_motion(kin, T0: np.ndarray, a: np.ndarray, sgn: float, xy, z_top: float) -> list:
    """One arc: clear the stack, then roll the cup while the hand is still moving.

    A roll that happens after the hand has stopped looks like the cup spins in mid-air.
    The turn stays near 28 cm, which is where this wrist can reach it, and it starts
    only after the cup is off to the side of the stack. The last centimetres are a
    straight drop the set-down finishes, because the wrist housing meets the table.
    """
    away = np.asarray(xy, float) - kin.mount[:2]
    away = away / max(float(np.linalg.norm(away)), 1e-6)
    R0 = T0[:3, :3].copy()
    p0 = T0[:3, 3].copy()
    z_clear = max(0.31, float(p0[2]) + 0.02)
    z_roll = 0.275
    phi_end = np.pi * 170.0 / 180.0
    poses = []

    def add(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(p, float)
        poses.append(T)

    # Clear the stack first. Turning while the cup is still in the nest knocks it out of the fingers.
    for z in np.linspace(float(p0[2]), z_clear, 4)[1:]:
        add(R0, [p0[0], p0[1], z])
    for s in (0.05, 0.09):
        add(R0, [p0[0] - away[0] * s, p0[1] - away[1] * s, z_clear])
    for z in np.linspace(z_clear, z_roll, 3)[1:]:
        add(R0, [p0[0] - away[0] * 0.09, p0[1] - away[1] * 0.09, z])
    # Roll at the height the wrist can reach, and keep sliding so the hand does not stop and spin.
    for i, phi in enumerate(np.linspace(0.0, phi_end, 14)[1:], start=1):
        u = i / 13.0
        s = 0.09 + 0.025 * u
        add(rot_about(a, sgn * phi) @ R0, [p0[0] - away[0] * s, p0[1] - away[1] * s, z_roll])
    end = poses[-1]
    for z in np.linspace(z_roll, 0.055, 4)[1:]:
        add(end[:3, :3], [end[0, 3], end[1, 3], z])
    return poses


def plan_pregrasp(ctx: Ctx) -> dict:
    """Turn the wrist to the flip attitude before the fingers touch the cup, and keep only attitudes whose rising roll lands flat."""
    n = ctx.bel.stack_n
    if n is None:
        raise SkillFault("stack height unknown: find it with the peck first")
    kin = ctx.kin["cup"]
    xy = np.asarray(ctx.bel.stack_xy, float)
    z = top_face_z(n) - TOP_GRASP_DEPTH
    q_now = ctx.q("cup")
    last = "no attitude reached"
    for ao, tilt in PREGRASP:
        T1, a, sgn = _flip_frame(kin, xy, z, ao, tilt)
        seeds = _flip_seeds(sgn)
        T_pre = T1.copy()
        T_pre[:3, 3] = T1[:3, 3] - a * 0.04 + np.array([0.0, 0.0, 0.02])
        T_hi = T1.copy()
        T_hi[:3, 3] = T1[:3, 3] - a * 0.03 + np.array([0.0, 0.0, 0.08])
        rp = kin.ik_multi(T_pre, seeds + [q_now])
        rh = kin.ik_multi(T_hi, seeds + ([rp.q] if rp.ok else []) + [q_now])
        rg = kin.ik_multi(T1, seeds + ([rp.q] if rp.ok else []))
        if not (rp.ok and rg.ok):
            last = (rp.reason or rg.reason or "grasp")[:80]
            continue
        z_top = min(0.285, z + 0.09)
        poses = flip_motion(kin, T1, a, sgn, xy, z_top)
        # drop flatten samples the wrist cannot reach; the descent to 6 cm is the part that must succeed
        kept, qprev = [], rg.q
        lean_note = ""
        for T in poses:
            r = kin.ik_multi(T, [qprev] + seeds)
            if not r.ok or (kept and float(np.max(np.abs(r.q - kept[-1]))) > 0.55):
                if len(kept) >= 16 + 3 + 4:          # roll, slide and descent are in; a later flatten sample missed
                    lean_note = r.reason if not r.ok else "joint jump"
                    break
                last = (r.reason if not r.ok else "joint jump")
                kept = []
                break
            kept.append(r.q)
            qprev = r.q
        if len(kept) < 20:
            continue
        try:
            app_q = [q_now] + ([rh.q] if rh.ok else []) + [rp.q]
            app = M.via_move(app_q, speed_scale=ctx.speed, min_T=1.0)
            ctx.check_traj("cup", app, grip_width=L.GRIPPER_MAX_OPEN_M)
            flip = M.via_move(kept, speed_scale=ctx.speed, min_T=3.2)
            ctx.check_traj("cup", flip, other_q=L.REST_Q, grip_width=0.055)
        except PlanError as exc:
            last = str(exc)[:120]
            continue
        return dict(ao=ao, tilt=tilt, sgn=sgn, a=a.copy(), grasp=T1, pre=T_pre,
                    q_hi=rh.q if rh.ok else None, q_pre=rp.q, z=z, z_top=z_top, n=n,
                    lean_note=lean_note, flip_qs=kept)
    raise SkillFault(f"no pre-rotated grasp that flips the cup flat on the way up ({last})")


def s_approach_stack(ctx: Ctx):
    plan = plan_pregrasp(ctx)
    ctx.scratch["pregrasp"] = plan
    ctx.scratch["stack_fr"] = dict(grasp=plan["grasp"], a=plan["a"], pre=plan["pre"])
    qs = [ctx.q("cup")] + ([plan["q_hi"]] if plan.get("q_hi") is not None else []) + [plan["q_pre"]]
    traj = M.via_move(qs, speed_scale=ctx.speed, min_T=1.0)
    ctx.check_traj("cup", traj, grip_width=L.GRIPPER_MAX_OPEN_M)
    yield from ctx.play("cup", traj)
    return {"msg": (f"wrist already turned {plan['ao']:+d} deg (the other way) and tilted {abs(plan['tilt']):.0f} deg down, "
                    f"before the fingers touch the cup; stack of {plan['n']}, TCP {plan['z'] * 100:.1f} cm")}


def s_slide_in_stack(ctx: Ctx):
    fr = ctx.scratch["stack_fr"]
    kin = ctx.kin["cup"]
    traj = M.cart_move(kin, [ctx.tcp("cup"), fr["grasp"]], ctx.q("cup"), v_cart=L.APPROACH_SPEED_M_S, speed_scale=ctx.speed)
    ctx.monitor("cup", True)
    a = fr["a"]
    done = yield from ctx.play("cup", traj, guard=lambda: float(-(ctx.mon["cup"].force @ a)) > 5.0)
    ctx.monitor("cup", False)
    if not done:
        raise SkillFault("fingers jammed on the top cup while sliding in (force above 5 N)")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    return {"msg": "fingers straddle the top cup's base band"}


def s_close_stack(ctx: Ctx):
    cap = L.FORCE_CAP_STACK_TOP_N
    res = yield from ctx.grip_close("cup", cap, rate=0.04)
    w = res["width"]
    if not res["stalled"]:
        raise SkillFault("jaws closed on nothing at the top of the stack")
    if not (0.048 < w < 0.066):
        raise SkillFault(f"jaws stalled at {fmt_mm(w)}; the top cup's base band is about 55 mm")
    tcp = ctx.tcp("cup")
    fr = ctx.scratch["stack_fr"]
    ctx.bel.found["stack_tcp"] = tcp[:3, 3].copy()
    ctx.bel.cup_in_hand = True
    ctx.scratch["top_hold_width"] = w
    return {"msg": f"jaws stalled at {fmt_mm(w)} with {res['force']:.1f} N of {cap:.0f} N: top cup gripped"}


# ---------------------------------------------------------------------------
# tap arm: pin the lower stack
# ---------------------------------------------------------------------------

def pin_frame(ctx: Ctx, az_deg: float, tilt_deg: float = 0.0):
    n = ctx.bel.stack_n
    xy = ctx.bel.stack_xy.copy()
    z = stack_rim_z(n) - PIN_BAND_BELOW_RIM
    az = np.radians(az_deg)
    t = np.radians(tilt_deg)
    # Negative tilt points the approach down, so the wrist housing (behind the fingers) stays above the mat.
    a = np.array([np.cos(az) * np.cos(t), np.sin(az) * np.cos(t), np.sin(t)])
    j = np.array([-np.sin(az), np.cos(az), 0.0])
    tcp = np.array([xy[0], xy[1], z]) + a * 0.004
    return a, j, tcp, z


def s_pin_approach(ctx: Ctx):
    kin = ctx.kin["tap"]
    q0 = ctx.q("tap")
    seeds = [np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.5, 1.2, 0.8, 0.6, 0.0, 0.0]),
             np.array([0.4, 1.0, 0.6, 0.2, 0.0, 0.0]), np.array([0.8, 1.6, 1.2, 0.4, -0.6, 0.4]), q0]
    z_pin = stack_rim_z(ctx.bel.stack_n) - PIN_BAND_BELOW_RIM
    # A level pinch below about 5 cm puts the housing in the mat. Try that first on a tall stack, then tilt down.
    tilts = (0.0, -20.0, -35.0, -50.0) if z_pin >= 0.05 else (-25.0, -40.0, -55.0, -15.0, 0.0)
    last = ""
    for az_deg in (60.0, 45.0, 75.0, 90.0, 30.0, 15.0, 0.0):
        for tilt in tilts:
            a, j, tcp, z = pin_frame(ctx, az_deg, tilt)
            T_g = frame_from(a, j, tcp)
            T_pre = frame_from(a, j, tcp - a * 0.06 + Z_UP * 0.004)
            T_hi = frame_from(a, j, tcp - a * 0.08 + Z_UP * max(0.12, 0.20 - float(tcp[2])))
            rp = kin.ik_multi(T_pre, seeds)
            rh = kin.ik_multi(T_hi, seeds + ([rp.q] if rp.ok else []))
            rg = kin.ik_multi(T_g, ([rp.q] if rp.ok else []) + seeds)
            if not (rp.ok and rh.ok and rg.ok):
                last = f"az {az_deg:.0f} tilt {tilt:.0f}: {(rp.reason or rh.reason or rg.reason or 'ik')[:80]}"
                continue
            if ctx.plant.pin_touches_free_cup(rg.q, 0.070):
                last = f"az {az_deg:.0f} tilt {tilt:.0f}: that pinch reaches the top cup"
                continue
            try:
                if tilt == 0.0:
                    traj = M.via_move([q0, rh.q, rp.q], speed_scale=ctx.speed, min_T=1.0)
                else:
                    traj = M.cart_move(kin, [kin.fk(q0), kin.fk(rh.q), T_pre], q0, v_cart=0.12, speed_scale=ctx.speed, min_T=1.2, max_joint_step=0.35)
                ctx.check_traj("tap", traj, grip_width=L.GRIPPER_MAX_OPEN_M)
            except PlanError as exc:
                last = f"az {az_deg:.0f} tilt {tilt:.0f}: {exc}"
                continue
            yield from ctx.grip_open("tap", L.GRIPPER_MAX_OPEN_M)
            yield from ctx.play("tap", traj)
            ctx.scratch["pin_fr"] = dict(grasp=T_g, pre=T_pre, a=a, q_pre=rp.q, az=az_deg, tilt=tilt)
            res = yield from _pin_close(ctx)
            if res is None:
                last = f"az {az_deg:.0f} tilt {tilt:.0f}: pinch was not on the band"
                yield from ctx.grip_open("tap", L.GRIPPER_MAX_OPEN_M)
                back = ctx.tcp("tap").copy()
                back[:3, 3] = back[:3, 3] - a * 0.05 + Z_UP * 0.06
                try:
                    retreat = M.cart_move(kin, [ctx.tcp("tap"), back], ctx.q("tap"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.5, max_joint_step=0.3)
                    ctx.check_traj("tap", retreat, grip_width=L.GRIPPER_MAX_OPEN_M)
                    yield from ctx.play("tap", retreat)
                except PlanError:
                    pass
                continue
            ctx.scratch["pin_done"] = res
            return {"msg": f"pads open, 6 cm from the lower stack (approach {az_deg:.0f} deg, tilt {tilt:.0f} deg)"}
    raise SkillFault(f"no collision-free way for the tap arm to reach the lower stack ({last})")


def _pin_close(ctx: Ctx):
    """Slide in and pinch. Returns the success message, or None if the stall width is not the lower stack's band."""
    fr = ctx.scratch["pin_fr"]
    kin = ctx.kin["tap"]
    traj = M.cart_move(kin, [ctx.tcp("tap"), fr["grasp"]], ctx.q("tap"), v_cart=L.APPROACH_SPEED_M_S, speed_scale=ctx.speed)
    ctx.monitor("tap", True)
    a = fr["a"]
    done = yield from ctx.play("tap", traj, guard=lambda: float(-(ctx.mon["tap"].force @ a)) > 5.0)
    if not done:
        ctx.monitor("tap", False)
        return None
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    cap = 26.0 if abs(float(fr.get("tilt", 0.0))) > 1.0 else L.FORCE_CAP_STACK_PIN_N
    res = yield from ctx.grip_close("tap", cap, rate=0.03)
    w = res["width"]
    ctx.monitor("tap", False)
    # 70 mm is the band. Wider than 73 mm is a pad on the rim or the mat, not a pinch that can hold the stack.
    if not res["stalled"] or not (0.066 < w < 0.073):
        return None
    return f"pinned: pads stalled at {fmt_mm(w)} with {res['force']:.1f} N of {L.FORCE_CAP_STACK_PIN_N:.0f} N"


def s_pin_hold(ctx: Ctx):
    if ctx.scratch.get("pin_done"):
        return {"msg": ctx.scratch["pin_done"]}
    fr = ctx.scratch["pin_fr"]
    kin = ctx.kin["tap"]
    traj = M.cart_move(kin, [ctx.tcp("tap"), fr["grasp"]], ctx.q("tap"), v_cart=L.APPROACH_SPEED_M_S, speed_scale=ctx.speed)
    ctx.monitor("tap", True)
    a = fr["a"]
    done = yield from ctx.play("tap", traj, guard=lambda: float(-(ctx.mon["tap"].force @ a)) > 5.0)
    if not done:
        ctx.monitor("tap", False)
        raise SkillFault("pads jammed on the lower stack while sliding in (force above 5 N)")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    res = yield from ctx.grip_close("tap", L.FORCE_CAP_STACK_PIN_N, rate=0.03)
    w = res["width"]
    ctx.monitor("tap", False)
    if not res["stalled"] or not (0.060 < w < 0.0765):
        raise SkillFault(f"pin pads stalled at {fmt_mm(w)}; the lower stack is about 70 mm wide")
    return {"msg": f"pinned: pads stalled at {fmt_mm(w)} with {res['force']:.1f} N of {L.FORCE_CAP_STACK_PIN_N:.0f} N"}


def s_pin_release(ctx: Ctx):
    yield from ctx.grip_open("tap", L.GRIPPER_MAX_OPEN_M)
    kin = ctx.kin["tap"]
    T0 = ctx.tcp("tap")
    a = T0[:3, 2]
    last = ""
    for back, up in ((0.04, 0.12), (0.05, 0.08), (0.07, 0.04), (0.07, 0.01)):
        T1 = T0.copy()
        T1[:3, 3] = T0[:3, 3] - a * back + Z_UP * up
        try:
            traj = M.cart_move(kin, [T0, T1], ctx.q("tap"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.7, max_joint_step=0.3)
            ctx.check_traj("tap", traj)
        except PlanError as exc:
            last = str(exc)
            continue
        yield from ctx.play("tap", traj)
        return {"msg": f"pads open, hand backed off the stack ({back * 100:.0f} cm back, {up * 100:.0f} cm up)"}
    raise SkillFault(f"the pin hand cannot back off without hitting the other arm ({last})")


# ---------------------------------------------------------------------------
# cup arm: wiggle the top cup free
# ---------------------------------------------------------------------------

def s_unstack(ctx: Ctx):
    """Wiggle to loosen the nested cups, then lift slowly and watch the pull force. The cups let go when it collapses.

    The wiggle's own inertia makes the force estimate unusable, so detection happens only during the slow lift."""
    kin = ctx.kin["cup"]
    n = ctx.bel.stack_n
    T0 = ctx.tcp("cup")
    # pull along the cup axis (hand x points from the base down into the nest), not world-vertical: a pre-tilted hand binds if it lifts crooked
    pull = -T0[:3, 0].copy()
    pull = pull / max(float(np.linalg.norm(pull)), 1e-9)
    if pull[2] < 0.0:
        pull = -pull
    u_grid = path_q = None
    z_grid = None
    for reach in (0.090, 0.050):
        zs = np.arange(-0.006, reach, 0.001)
        poses = []
        for dz in (zs[0], zs[-1]):
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + pull * dz
            poses.append(T)
        try:
            u_grid, path_q, length = M.cart_path(kin, poses, ctx.q("cup"), step_m=0.001, max_joint_step=0.2)
            z_grid = zs[0] + u_grid * (zs[-1] - zs[0])
            break
        except PlanError:
            if reach == 0.050:
                raise

    def q_at(z: float):
        zc = float(np.clip(z, z_grid[0], z_grid[-1]))
        q = np.array([np.interp(zc, z_grid, path_q[:, j]) for j in range(6)])
        dq = np.array([np.interp(zc + 0.0005, z_grid, path_q[:, j]) - np.interp(zc - 0.0005, z_grid, path_q[:, j]) for j in range(6)]) / 0.001
        return q, dq

    dt = CTRL_STEPS * ctx.plant.dt
    # The pin arm's position servo yields under the tether and the lower stack creeps up with the cup,
    # so the bond never loads. Put the pin joints back on the pose that closed every tick of the pull.
    tap_idx = ctx.plant.arm["tap"]
    pin_q = ctx.q("tap").copy()

    def hold_pin() -> None:
        ctx.plant.data.qpos[tap_idx.qpos] = pin_q
        ctx.plant.data.qvel[tap_idx.dof] = 0.0
        ctx.plant.command_arm("tap", pin_q)

    ctx.monitor("cup", True)
    base = Baseline()
    for _ in range(int(0.45 / dt)):
        base.add(ctx.mon["cup"].force)
        ctx.hold("cup")
        yield
    f0 = float(base.stats()[0] @ pull)
    z_now = 0.0
    freed = False
    peak = 0.0
    attempts = 0
    t_total = 0.0
    short = (ctx.bel.stack_n or 8) <= 4
    squeeze = 28.0 if (ctx.plant.material == "plastic" or short) else L.FORCE_CAP_STACK_TOP_N
    while attempts < 4 and not freed:
        attempts += 1
        if short and attempts > 1 and z_now > 0.012:
            # A short stack only has 1-2 cm of band in the pin pads. Step back down into that
            # band before the next wiggle so the lower cups are not hauled out of the pinch.
            z_back = max(0.008, z_now - 0.014)
            for zb in np.linspace(z_now, z_back, 40):
                q, _ = q_at(float(zb))
                ctx.q_cmd["cup"] = q
                ctx.plant.command_arm("cup", q)
                hold_pin()
                yield
            z_now = z_back
        yield from ctx.grip_close("cup", squeeze, rate=0.06)
        ctx.scratch["top_hold_width"] = ctx.width("cup")
        # 1. wiggle in place: ramp the amplitude in, hold, ramp out
        amp = 0.0020
        t = 0.0
        while t < 2.0:
            ramp = min(t / 0.5, 1.0) * min((2.0 - t) / 0.4, 1.0)
            phase = 2.0 * np.pi * 2.4 * t
            z_cmd = z_now + amp * ramp * np.sin(phase)
            q, dq = q_at(z_cmd)
            qd = dq * (amp * ramp * 2.0 * np.pi * 2.4 * np.cos(phase))
            ctx.q_cmd["cup"] = q
            ctx.plant.command_arm("cup", q, qd)
            t += dt
            t_total += dt
            yield
        yield from ctx.wait(0.2)
        if ctx.width("cup") > ctx.scratch.get("top_hold_width", 0.05) + 0.0012:
            yield from ctx.grip_close("cup", squeeze, rate=0.05)
            ctx.scratch["top_hold_width"] = ctx.width("cup")
        # 2. slow quasi-static lift, watch the pull force
        z_start = z_now
        v = 0.005 if ctx.plant.material == "plastic" else 0.01
        lift = 0.028
        t = 0.0
        peak = 0.0
        low = 0
        smooth = 0.0
        while t < lift / v:
            z_cmd = z_start + v * t
            q, dq = q_at(z_cmd)
            ctx.q_cmd["cup"] = q
            ctx.plant.command_arm("cup", q, dq * v)
            hold_pin()
            load = -(float(ctx.mon["cup"].force @ pull) - f0)
            smooth += 0.12 * (load - smooth)
            peak = max(peak, smooth)
            # A real separation spikes near the bond limit and then collapses. A flat load is the lower stack
            # still coming along, not a cup that is already free.
            if peak > 2.2 and smooth < min(0.35 * peak, 0.75):
                low += 1
            else:
                low = 0
            if low >= 25:
                freed = True
                z_now = z_cmd
                break
            t += dt
            t_total += dt
            yield
        else:
            z_now = z_start + lift
            # The wiggle can break the bond before this pull. What is left is the cup's own weight.
            w_now = ctx.width("cup")
            held = (0.045 < w_now < ctx.scratch.get("top_hold_width", 0.05) + 0.0025
                    and ctx.mon["cup"].grip_force > 2.5)
            if peak < (0.85 if short else 1.5) and held:
                freed = True
        if not freed:
            ctx.event(f"unstack attempt {attempts}: pull peaked at {peak:.1f} N and did not collapse, wiggling again")
    ctx.scratch["unstack_peak_load"] = peak
    rest_z = float(ctx.plant.data.xpos[ctx.plant.body[ctx.plant.active_rest]][2])
    if freed and rest_z > L.MAT_SURFACE_Z + 0.06:
        raise SkillFault(f"the lower stack rose with the cup (stack base at {rest_z * 100:.1f} cm)")
    if not freed:
        raise SkillFault(f"the top cup would not come free (peak pull {peak:.1f} N after {attempts} wiggle-and-pull attempts)")
    # Re-seat the pinch before any further move. A hard break can leave the cup sitting in jaws that have just unloaded.
    yield from ctx.grip_close("cup", squeeze, rate=0.08)
    # Lift clear of the pin hand before it backs off, or the retreat knocks the cup out of the fingers.
    T1 = ctx.tcp("cup")
    z_here = float(T1[2, 3])
    z_clear = max(z_here + 0.04, 0.32)
    for z_goal in (z_clear, z_here + 0.03):
        T2 = T1.copy()
        T2[2, 3] = z_goal
        try:
            traj = M.cart_move(kin, [T1, T2], ctx.q("cup"), v_cart=0.04, speed_scale=ctx.speed, min_T=0.5, max_joint_step=0.25)
            ctx.check_traj("cup", traj, grip_width=ctx.width("cup"))
            yield from ctx.play("cup", traj, on_tick=lambda _k: hold_pin())
            break
        except PlanError:
            continue
    samples = []
    for _ in range(int(0.2 / dt)):
        samples.append(float(ctx.mon["cup"].force @ pull))
        ctx.hold("cup")
        yield
    grip_now = ctx.mon["cup"].grip_force
    ctx.monitor("cup", False)
    # Once the bond breaks, the pinch reading falls even though the jaws are still on the cup.
    # An empty hand reads about nothing; a cup that is still there stays above a couple of newtons.
    if grip_now < 3.0 or ctx.width("cup") > ctx.scratch["top_hold_width"] + 0.003:
        raise SkillFault(f"the cup slipped out of the jaws during the pull (grip {grip_now:.1f} N, jaws {fmt_mm(ctx.width('cup'))})")
    load_end = -(float(np.mean(samples)) - f0)
    if load_end > 8.0:
        raise SkillFault(f"the lower cups came up with it (extra load {load_end:.2f} N)")
    return {"msg": f"separated after {attempts} wiggle-and-pull attempt(s), {t_total:.1f} s; peak pull {peak:.1f} N, load now {max(load_end, 0):.2f} N (one cup)"}


# ---------------------------------------------------------------------------
# cup arm: carry high, roll, lay the cup on its side
# ---------------------------------------------------------------------------

HANDOFF_B = np.array([0.44, -0.10])             # where the base end of the sideways cup goes (world xy)
HANDOFF_AZ_DEG = -110.0                         # the lying cup points this way (mouth toward the tap arm)
HAND_PITCH_DEG = -45.0                          # hand tilted down 45 deg about the cup axis; the cup does not care


def lying_axis(b_h: np.ndarray) -> np.ndarray:
    """Cup axis (base to rim) of a cone resting on a generatrix: the wide rim end is higher by the taper angle."""
    t = np.arctan(L.CUP_TAPER)
    return np.array([np.cos(t) * b_h[0], np.cos(t) * b_h[1], np.sin(t)])


def lying_hand_frame(pos: np.ndarray, b_h: np.ndarray, psi_deg: float = HAND_PITCH_DEG) -> np.ndarray:
    """Hand frame that holds the cup along the lying axis with the fingers pointing down-forward.

    Rotating the hand about the cup's own axis does not change the grasp, so the pitch is free: it keeps the wide
    housing plate above the mat and the hand reachable."""
    psi = np.radians(psi_deg)
    b = lying_axis(b_h)
    eh = np.cross(Z_UP, b)
    eh /= np.linalg.norm(eh)
    u0 = np.cos(psi) * eh + np.sin(psi) * Z_UP
    z_s = u0 - (u0 @ b) * b
    z_s /= np.linalg.norm(z_s)
    y_s = np.cross(z_s, b)
    return frame_from(z_s, y_s, pos)


def s_carry_roll(ctx: Ctx):
    """Carry the hanging cup high to the handoff zone and swing it horizontal by rotating the hand about its own axes."""
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    b_hat = np.array([np.cos(np.radians(HANDOFF_AZ_DEG)), np.sin(np.radians(HANDOFF_AZ_DEG)), 0.0])
    B = HANDOFF_B
    tcp_xy = B + 0.008 * b_hat[:2]
    z_hi = 0.30
    T_hi = hold_frame_at(T0, np.array([tcp_xy[0], tcp_xy[1], z_hi + 0.03]))
    T_rolled = lying_hand_frame(np.array([tcp_xy[0], tcp_xy[1], z_hi]), b_hat)
    traj1 = M.cart_move(kin, [T0, T_hi], ctx.q("cup"), v_cart=0.15, speed_scale=ctx.speed, min_T=1.0)
    ctx.check_traj("cup", traj1)
    yield from ctx.play("cup", traj1)
    traj2 = M.cart_move(kin, [ctx.tcp("cup"), T_rolled], ctx.q("cup"), v_cart=0.05, speed_scale=ctx.speed, min_T=1.8, max_joint_step=0.4)
    ctx.check_traj("cup", traj2)
    yield from ctx.play("cup", traj2)
    ctx.scratch["handoff"] = dict(b_hat=b_hat, B=B.copy(), tcp_xy=tcp_xy)
    if ctx.mon["cup"].grip_force < 0.5 * L.FORCE_CAP_STACK_TOP_N:
        raise SkillFault(f"the cup is no longer gripped after the swing (grip {ctx.mon['cup'].grip_force:.1f} N)")
    return {"msg": f"cup swung horizontal, hand pitched {abs(HAND_PITCH_DEG):.0f} deg; it points {HANDOFF_AZ_DEG:.0f} deg toward the tap arm"}


def s_place_sideways(ctx: Ctx):
    kin = ctx.kin["cup"]
    ho = ctx.scratch["handoff"]
    T0 = ctx.tcp("cup")
    z_slow = 0.075
    z_goal = 0.0318            # base-band axis height of a cone lying on a generatrix is 3 + 27.5 mm; lower pad clears the mat
    T_a = T0.copy()
    T_a[2, 3] = z_slow
    T_b = T0.copy()
    T_b[2, 3] = z_goal
    fast = M.cart_move(kin, [T0, T_a], ctx.q("cup"), v_cart=0.12, speed_scale=ctx.speed, min_T=0.8)
    ctx.check_traj("cup", fast)
    yield from ctx.play("cup", fast)
    slow = M.cart_move(kin, [ctx.tcp("cup"), T_b], ctx.q("cup"), v_cart=0.015, speed_scale=ctx.speed, min_T=1.0)
    ctx.check_traj("cup", slow)
    ctx.monitor("cup", True)
    state = {"n": 0}

    def on_tick(k: int) -> None:
        state["z_cmd"] = float(kin.fk(slow.q[k])[2, 3])

    def guard() -> bool:
        m = ctx.mon["cup"]
        z_now = float(m.tcp[2, 3])
        if z_now > z_goal + 0.022:
            return False
        lag = z_now - state.get("z_cmd", z_now)
        if float(m.delta_force(60)[2]) > 0.9 or lag > 0.004:
            state["n"] += 1
        else:
            state["n"] = 0
        return state["n"] >= 5

    done = yield from ctx.play("cup", slow, guard=guard, on_tick=on_tick)
    z_end = float(ctx.tcp("cup")[2, 3])
    yield from ctx.wait(0.25)
    yield from ctx.grip_open("cup", 0.064, rate=0.05)
    yield from ctx.wait(0.7)
    ctx.monitor("cup", False)
    tcp = ctx.tcp("cup")
    b_hat = tcp[:3, 0].copy()
    b_hat[2] = 0.0
    b_hat /= np.linalg.norm(b_hat)
    B = tcp[:2, 3] - 0.008 * b_hat[:2]
    ctx.bel.sideways_cup = dict(B=B, b_hat=b_hat[:2].copy())
    ctx.bel.cup_in_hand = False
    where = "touched the mat" if not done else "reached the place height without a contact spike"
    return {"msg": f"rim end {where} at TCP z {z_end * 1000:.0f} mm, jaws opened; base end at ({B[0]:.3f}, {B[1]:.3f})"}


def s_lift_off_cup(ctx: Ctx):
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    a = T0[:3, 2]
    T1 = T0.copy()
    T1[:3, 3] = T0[:3, 3] - a * 0.05 + Z_UP * 0.03
    T2 = T1.copy()
    T2[2, 3] += 0.10
    traj = M.cart_move(kin, [T0, T1, T2], ctx.q("cup"), v_cart=0.10, speed_scale=ctx.speed, min_T=1.2)
    ctx.check_traj("cup", traj, grip_width=ctx.width("cup"))
    yield from ctx.play("cup", traj)
    yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M)
    return {"msg": "hand backed along its fingers and lifted away from the lying cup"}


# ---------------------------------------------------------------------------
# tap arm: pinch the rim, tip the cup upright about its base edge
# ---------------------------------------------------------------------------

def rim_frames(ctx: Ctx):
    sc = ctx.bel.sideways_cup
    B, b = sc["B"], np.append(sc["b_hat"], 0.0)
    rim_center = np.array([B[0], B[1], L.MAT_SURFACE_Z + L.CUP_RIM_RADIUS]) + L.CUP_HEIGHT * b
    rim_top = rim_center + Z_UP * L.CUP_RIM_RADIUS
    a_t = -b
    j = Z_UP
    tcp = rim_top + a_t * 0.010 - Z_UP * 0.0009
    T_g = frame_from(a_t, j, tcp)
    T_pre = frame_from(a_t, j, tcp - a_t * 0.065 + Z_UP * 0.004)
    T_hi = frame_from(a_t, j, tcp - a_t * 0.065 + Z_UP * 0.07)
    P = np.array([B[0], B[1], L.MAT_SURFACE_Z])
    axis = np.cross(b, Z_UP)
    axis /= np.linalg.norm(axis)
    return dict(T_g=T_g, T_pre=T_pre, T_hi=T_hi, P=P, axis=axis, b=b, a_t=a_t, rim_top=rim_top)


def s_rim_approach(ctx: Ctx):
    kin = ctx.kin["tap"]
    fr = rim_frames(ctx)
    q0 = ctx.q("tap")
    seeds = [np.array([0.6, 1.8, 1.3, 0.9, 0.0, 0.0]), np.array([0.5, 1.2, 0.8, 0.6, 0.0, 0.0]), np.array([0.3, 2.0, 1.6, 1.4, 0.2, 0.0]), q0]
    rp = kin.ik_multi(fr["T_pre"], seeds)
    rh = kin.ik_multi(fr["T_hi"], seeds + [rp.q])
    if not (rp.ok and rh.ok):
        raise SkillFault(f"rim approach unreachable ({rp.reason or rh.reason})")
    yield from ctx.grip_open("tap", 0.026)
    traj = M.via_move([q0, rh.q, rp.q], speed_scale=ctx.speed, min_T=1.2)
    ctx.check_traj("tap", traj, grip_width=0.026)
    yield from ctx.play("tap", traj)
    ctx.scratch["rim_fr"] = fr
    return {"msg": "pads 26 mm apart, 6.5 cm in front of the lying cup's mouth, level with its rim top"}


def s_rim_grab(ctx: Ctx):
    fr = ctx.scratch["rim_fr"]
    kin = ctx.kin["tap"]
    traj = M.cart_move(kin, [ctx.tcp("tap"), fr["T_g"]], ctx.q("tap"), v_cart=0.025, speed_scale=ctx.speed, min_T=1.0)
    ctx.monitor("tap", True)
    a = fr["a_t"]
    done = yield from ctx.play("tap", traj, guard=lambda: float(-(ctx.mon["tap"].force @ a)) > 4.0)
    if not done:
        ctx.monitor("tap", False)
        raise SkillFault("pads jammed against the cup while reaching for the rim")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    res = yield from ctx.grip_close("tap", L.FORCE_CAP_RIM_AND_LEVER_N, rate=0.02, min_width=0.0009)
    w = res["width"]
    if not res["stalled"] or not (0.0009 < w < 0.012):
        ctx.monitor("tap", False)
        raise SkillFault(f"pads stalled at {fmt_mm(w)}; a cup wall is about 2 mm")
    ctx.scratch["rim_pinch_tcp"] = ctx.tcp("tap")
    return {"msg": f"pinching the rim wall: pads stalled at {fmt_mm(w)} with {res['force']:.1f} N"}


def s_flip_up(ctx: Ctx):
    kin = ctx.kin["tap"]
    fr = ctx.scratch["rim_fr"]
    T_pinch = ctx.scratch["rim_pinch_tcp"]
    P, axis = fr["P"], fr["axis"]
    thetas = np.radians(np.arange(0.0, FLIP_RELEASE_DEG + 0.1, 3.0))
    poses = []
    for th in thetas:
        R = rot_about(axis, th)
        T = np.eye(4)
        T[:3, :3] = R @ T_pinch[:3, :3]
        T[:3, 3] = P + R @ (T_pinch[:3, 3] - P)
        poses.append(T)
    try:
        traj = M.cart_move(kin, [ctx.tcp("tap")] + poses, ctx.q("tap"), v_cart=0.05, speed_scale=ctx.speed, min_T=3.2, step_m=0.003,
                           max_joint_step=0.2)
    except PlanError as exc:
        ctx.monitor("tap", False)
        raise SkillFault(f"cannot swing the cup upright about its base edge: {exc}")
    ctx.check_traj("tap", traj, grip_width=ctx.width("tap"))
    w0 = ctx.width("tap")
    yield from ctx.play("tap", traj)
    if ctx.width("tap") > w0 + 0.004:
        raise SkillFault("the rim slipped out of the pads during the swing")
    yield from ctx.wait(0.4)
    ctx.monitor("tap", False)
    # The hand holds the cup rigidly, so the measured hand pose gives the cup pose: no guessing where it landed.
    T_end = ctx.tcp("tap")
    sc = ctx.bel.sideways_cup
    b_h = np.append(sc["b_hat"], 0.0)
    rim_center0 = fr["rim_top"] - Z_UP * L.CUP_RIM_RADIUS
    v_hand = np.linalg.inv(T_pinch) @ np.append(rim_center0, 1.0)
    rim_center_end = (T_end @ v_hand)[:3]
    # The hand pose gives the rim centre (the pinched wall sits on it). How far the cup leans, and so where its base ends up
    # standing, is only known to about 1.5 cm, so the re-grasp below approaches across that uncertainty and lets the
    # funnel pads and the closing jaws centre the cup.
    axis_end = np.array([0.0, 0.0, 1.0])
    base_center = rim_center_end - L.CUP_HEIGHT * axis_end
    stand_xy = rim_center_end[:2] - 0.034 * b_h[:2]
    ctx.bel.flip_result = dict(axis=axis_end, base_center=base_center, rim_center=rim_center_end, stand_xy=stand_xy, b_hat=b_h[:2].copy())
    return {"msg": f"cup swung {FLIP_RELEASE_DEG:.0f} deg about its base edge; rim centre at ({rim_center_end[0]:.3f}, {rim_center_end[1]:.3f}), it will stand near ({stand_xy[0]:.3f}, {stand_xy[1]:.3f})"}


def s_flip_release(ctx: Ctx):
    """Open the pads, then pitch the hand about its own jaw axis while lifting: the pads slide up the wall without
    side loads and the wrist leaves its joint limit."""
    yield from ctx.grip_open("tap", 0.040)
    yield from ctx.wait(0.8)
    kin = ctx.kin["tap"]
    T0 = ctx.tcp("tap")
    q0 = ctx.q("tap")
    jaw = T0[:3, 1]
    last = ""
    for pitch in (25.0, 40.0, 15.0, 0.0):
        for lift in (0.05, 0.08, 0.03):
            for sgn in ((1.0, -1.0) if pitch > 0 else (1.0,)):
                R = rot_about(jaw, sgn * np.radians(pitch))
                T1 = np.eye(4)
                T1[:3, :3] = R @ T0[:3, :3]
                T1[:3, 3] = T0[:3, 3] + Z_UP * lift
                try:
                    traj = M.cart_move(kin, [T0, T1], q0, v_cart=0.04, speed_scale=ctx.speed, min_T=1.4, max_joint_step=0.2)
                    ctx.check_traj("tap", traj)
                except PlanError as exc:
                    last = str(exc)
                    continue
                yield from ctx.play("tap", traj)
                fr_res = ctx.bel.flip_result
                ctx.bel.cup_xy = np.array(fr_res["stand_xy"], dtype=float)
                bh = fr_res["b_hat"]
                ahead = np.array([bh[1], -bh[0]])                       # across the lying direction, toward +x
                if ahead[0] < 0:
                    ahead = -ahead
                ctx.scratch["regrasp_az"] = float(np.arctan2(ahead[1], ahead[0]))
                return {"msg": f"pads open; hand pitched {sgn * pitch:+.0f} deg about the jaw axis and lifted {lift * 100:.0f} cm off the wall"}
    raise SkillFault(f"the tap hand cannot leave the cup without touching it: {last}")


# ---------------------------------------------------------------------------
# cup arm: take the cup from the tap arm while it is still held tilted
# ---------------------------------------------------------------------------

def catch_frames(ctx: Ctx):
    fr = ctx.bel.flip_result
    axis = fr["axis"]
    x_s = -axis                                              # from the rim toward the base, like a standing-cup grasp
    kin = ctx.kin["cup"]
    base = fr["base_center"]
    h = base[:2] - kin.mount[:2]
    az = np.arctan2(h[1], h[0]) - np.radians(9.0)
    hz = np.array([np.cos(az), np.sin(az), 0.0])
    z_s = hz - (hz @ x_s) * x_s
    z_s /= np.linalg.norm(z_s)
    y_s = np.cross(z_s, x_s)
    grasp_pt = base + axis * HOLD_DZ
    T_g = frame_from(z_s, y_s, grasp_pt + z_s * 0.004)
    T_pre = frame_from(z_s, y_s, grasp_pt - z_s * 0.05 + Z_UP * 0.012)
    T_hi = frame_from(z_s, y_s, grasp_pt - z_s * 0.05 + Z_UP * 0.085)
    return T_g, T_pre, T_hi, z_s


def s_catch_approach(ctx: Ctx):
    kin = ctx.kin["cup"]
    T_g, T_pre, T_hi, z_s = catch_frames(ctx)
    q0 = ctx.q("cup")
    seeds = [L.REAL_L1_CUP_WAYPOINTS["paper_hold"][:6], np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0]), np.array([0.3, 1.2, 0.9, 0.5, 0.0, 0.0]), q0]
    rp, rh, rg = kin.ik_multi(T_pre, seeds), None, None
    rh = kin.ik_multi(T_hi, seeds + [rp.q])
    rg = kin.ik_multi(T_g, [rp.q] + seeds)
    if not (rp.ok and rh.ok and rg.ok):
        raise SkillFault(f"cannot reach the held cup from the side ({rp.reason or rh.reason or rg.reason})")
    yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M)
    traj = M.via_move([q0, rh.q, rp.q], speed_scale=ctx.speed, min_T=1.2)
    ctx.check_traj("cup", traj, grip_width=L.GRIPPER_MAX_OPEN_M)
    yield from ctx.play("cup", traj)
    ctx.scratch["catch"] = dict(T_g=T_g, T_pre=T_pre, a=z_s, q_pre=rp.q)
    return {"msg": "cup arm 5 cm beside the held cup, level with its middle"}


def s_catch_slide(ctx: Ctx):
    fr = ctx.scratch["catch"]
    kin = ctx.kin["cup"]
    traj = M.cart_move(kin, [ctx.tcp("cup"), fr["T_g"]], ctx.q("cup"), v_cart=L.APPROACH_SPEED_M_S, speed_scale=ctx.speed, min_T=1.0)
    ctx.monitor("cup", True)
    a = fr["a"]
    done = yield from ctx.play("cup", traj, guard=lambda: float(-(ctx.mon["cup"].force @ a)) > 5.0)
    ctx.monitor("cup", False)
    if not done:
        raise SkillFault("fingers jammed on the held cup while sliding in")
    yield from ctx.wait(L.CONTACT_PAUSE_S)
    return {"msg": "fingers straddle the held cup's wall"}


def s_catch_close(ctx: Ctx):
    cap = L.FORCE_CAP_PLASTIC_CUP_N if ctx.plant.material == "plastic" else L.FORCE_CAP_PAPER_CUP_N
    res = yield from ctx.grip_close("cup", cap, rate=0.05)
    w = res["width"]
    if not res["stalled"] or not (0.056 < w < 0.075):
        raise SkillFault(f"jaws stalled at {fmt_mm(w)}; expected a cup wall 57-74 mm wide")
    ctx.bel.cup_in_hand = True
    return {"msg": f"cup arm has the cup: jaws stalled at {fmt_mm(w)} with {res['force']:.1f} N"}


def s_straighten_lift(ctx: Ctx, dz: float = 0.05):
    """Lift while turning the hand so the cup ends vertical (x axis of the hand straight down)."""
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    x_s = -Z_UP
    z_s = T0[:3, 2] - (T0[:3, 2] @ x_s) * x_s
    z_s /= np.linalg.norm(z_s)
    y_s = np.cross(z_s, x_s)
    T1 = frame_from(z_s, y_s, T0[:3, 3] + Z_UP * dz)
    traj = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.06, speed_scale=ctx.speed, min_T=1.0)
    ctx.check_traj("cup", traj, grip_width=ctx.width("cup"))
    w0 = ctx.width("cup")
    yield from ctx.play("cup", traj)
    if ctx.width("cup") > w0 + 0.0025:
        raise SkillFault("cup slipped in the jaws during the lift")
    return {"msg": f"cup lifted {dz * 100:.0f} cm and straightened upright in the jaws"}





# ---------------------------------------------------------------------------
# One arm, one motion: flip the cup in the air with the wrist roll
# ---------------------------------------------------------------------------
# Hold the inverted cup by its base band. Tilt the hand 20-30 deg down and turn it 45-60 deg off the arm's radial line: in that
# attitude the wrist is aligned so that J6 alone sweeps from about -90 deg to +90 deg, which rolls the cup half a turn about the
# approach axis (that axis is perpendicular to the cup's axis, so the cup turns over without moving). No second arm, no laying
# the cup on the mat, no rim grab. The cup comes out upright but leaning ~20 deg, is lowered until its base edge touches the mat,
# and released: it settles upright on its base.

FLIP_CANDIDATES = [
    ((0.36, 0.05), 0.34, 60, -20), ((0.36, 0.05), 0.34, -60, -20), ((0.42, 0.10), 0.34, 60, -20), ((0.42, 0.10), 0.34, -60, -20),
    ((0.33, 0.15), 0.34, 45, -30), ((0.33, 0.15), 0.34, -45, -30), ((0.36, 0.05), 0.30, 60, -30), ((0.36, 0.05), 0.30, -60, -30),
    ((0.40, -0.06), 0.34, 60, -20), ((0.40, -0.06), 0.34, -60, -20), ((0.38, -0.03), 0.34, 60, -20), ((0.42, -0.02), 0.34, 60, -20),
]
FLIP_LOWER_Z = (0.22, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025)


def _flip_frame(kin, xy, z, az_off_deg, tilt_deg):
    base = kin.mount[:2]
    radial = np.asarray(xy, float) - base
    radial = radial / np.linalg.norm(radial)
    ao = np.radians(az_off_deg)
    h = np.array([np.cos(ao) * radial[0] - np.sin(ao) * radial[1], np.sin(ao) * radial[0] + np.cos(ao) * radial[1], 0.0])
    t = np.radians(tilt_deg)
    a = np.array([np.cos(t) * h[0], np.cos(t) * h[1], np.sin(t)])
    j = np.cross([0.0, 0.0, 1.0], a)
    j /= np.linalg.norm(j)
    return frame_from(a, j, [xy[0], xy[1], z]), a, (1.0 if az_off_deg > 0 else -1.0)


def plan_flip(ctx: Ctx) -> dict:
    kin = ctx.kin["cup"]
    q_now = ctx.q("cup")
    T_now = ctx.tcp("cup")
    for xy, z, ao, tilt in FLIP_CANDIDATES:
        T1, a, sgn = _flip_frame(kin, xy, z, ao, tilt)
        seeds = [np.array([0.0, 1.2, 1.1, 0.5, 0.0, -1.6 * sgn]), np.array([0.3, 1.0, 1.0, 0.8, 0.6, -1.6 * sgn]),
                 np.array([0.0, 1.6, 1.1, 0.7, 0.0, -1.5 * sgn]), np.array([0.2, 1.0, 0.5, 0.0, 0.0, -1.5 * sgn])]
        qs, qprev, ok = [], None, True
        for phi in np.radians(np.arange(0, 181, 15)):
            T = np.eye(4)
            T[:3, :3] = rot_about(a, sgn * phi) @ T1[:3, :3]
            T[:3, 3] = T1[:3, 3]
            r = kin.ik_multi(T, seeds + ([qprev] if qprev is not None else []))
            if not r.ok:
                ok = False
                break
            qprev = r.q
            qs.append(r.q)
        if not ok:
            continue
        T2 = np.eye(4)
        T2[:3, :3] = rot_about(a, sgn * np.pi) @ T1[:3, :3]
        T2[:3, 3] = T1[:3, 3]
        ql, qp = [], qs[-1]
        for zz in FLIP_LOWER_Z:
            T = T2.copy()
            T[2, 3] = zz
            r = kin.ik_multi(T, [qp])
            if not r.ok:
                ok = False
                break
            qp = r.q
            ql.append(r.q)
        if not ok:
            continue
        try:
            # first straight up with the hand as it is, then a smooth joint-space move to the roll-ready attitude
            T_up = T_now.copy()
            T_up[2, 3] = max(float(T_now[2, 3]), z + 0.02)
            up = M.cart_move(kin, [T_now, T_up], q_now, v_cart=0.08, speed_scale=ctx.speed, min_T=0.6)
            ctx.check_traj("cup", up)
            transit = M.via_move([up.q[-1], qs[0]], speed_scale=ctx.speed, min_T=1.8)
            ctx.check_traj("cup", transit)
            roll = M.via_move(qs, speed_scale=ctx.speed, min_T=2.6)
            ctx.check_traj("cup", roll)
            lower = M.via_move([qs[-1]] + ql, speed_scale=ctx.speed, min_T=4.5)
            ctx.check_traj("cup", lower)
        except PlanError:
            continue
        return dict(up=up, transit=transit, roll=roll, lower=lower, spot=xy, z=z, az_off=ao, tilt=tilt, q_roll_end=qs[-1], q_lower=ql)
    raise SkillFault("no reachable, collision-free attitude for the in-air flip")


def s_flip_carry(ctx: Ctx):
    plan = plan_flip(ctx)
    ctx.scratch["flip"] = plan
    yield from ctx.play("cup", plan["up"])
    yield from ctx.play("cup", plan["transit"])
    if ctx.mon["cup"].grip_force < 0.5 * L.FORCE_CAP_STACK_TOP_N:
        raise SkillFault(f"the cup is no longer gripped after the carry (grip {ctx.mon['cup'].grip_force:.1f} N)")
    return {"msg": f"cup held at ({plan['spot'][0]:.2f}, {plan['spot'][1]:.2f}), {plan['z'] * 100:.0f} cm up, hand turned {plan['az_off']:+d} deg and tilted {abs(plan['tilt']):.0f} deg: wrist aligned for the roll"}


def s_flip_roll(ctx: Ctx):
    plan = ctx.scratch["flip"]
    w0 = ctx.width("cup")
    j6_0 = float(ctx.q("cup")[5])
    yield from ctx.play("cup", plan["roll"])
    yield from ctx.wait(0.3)
    if ctx.mon["cup"].grip_force < 0.4 * L.FORCE_CAP_STACK_TOP_N or ctx.width("cup") > w0 + 0.003:
        raise SkillFault(f"the cup slipped in the jaws during the roll (grip {ctx.mon['cup'].grip_force:.1f} N)")
    return {"msg": f"wrist rolled {np.degrees(abs(float(ctx.q('cup')[5]) - j6_0)):.0f} deg in one motion (J6 alone): the cup is upright"}


def s_flip_lift(ctx: Ctx):
    """The wrist was turned before the grasp. Rising from here rolls the cup the other way and finishes with it upright and flat."""
    plan = ctx.scratch.get("pregrasp")
    if plan is None:
        raise SkillFault("the wrist was not turned before the grasp")
    kin = ctx.kin["cup"]
    w0 = ctx.width("cup")
    j6_0 = float(ctx.q("cup")[5])
    # The pull leaves the hand a few millimetres off the line that was solved. That offset puts the
    # top of the roll just outside the workspace, so step back onto the grasp line first.
    T = ctx.tcp("cup").copy()
    if float(np.linalg.norm(T[:2, 3] - plan["grasp"][:2, 3])) > 0.002:
        Tc = T.copy()
        Tc[:2, 3] = plan["grasp"][:2, 3]
        try:
            corr = M.cart_move(kin, [T, Tc], ctx.q("cup"), v_cart=0.02, speed_scale=ctx.speed,
                               min_T=0.4, max_joint_step=0.2)
            ctx.check_traj("cup", corr, other_q=L.REST_Q, grip_width=w0)
            ctx.monitor("cup", True)
            yield from ctx.play("cup", corr)
        except PlanError:
            pass
    # Rebuild the roll on the orientation that was solved before the grasp. Tracking during the
    # pull leaves the wrist a few degrees off that, and the off-axis roll falls out of reach.
    T0 = ctx.tcp("cup").copy()
    T0[:3, :3] = plan["grasp"][:3, :3]
    T0[:2, 3] = plan["grasp"][:2, 3]
    poses = flip_motion(kin, T0, plan["a"], plan["sgn"], ctx.bel.stack_xy, plan["z_top"])
    seeds = _flip_seeds(plan["sgn"])
    kept, qprev, missed = [], ctx.q("cup"), 0
    last_fail = "none"
    for i, Tp in enumerate(poses):
        r = kin.ik_multi(Tp, [qprev] + seeds)
        jumped = bool(kept) and r.ok and float(np.max(np.abs(r.q - kept[-1]))) > 0.55
        if not r.ok or jumped:
            missed += 1
            last_fail = f"pose {i} z {Tp[2, 3]:.3f} x {Tp[2, 0]:+.2f} " + (r.reason if not r.ok else "joint jump")
            if missed > 8:
                break
            continue
        missed = 0
        kept.append(r.q)
        qprev = r.q
    # The last centimetres of the descent put the wrist housing in the table. Keep the upright
    # prefix that stays clear; the set-down finishes the lower.
    other = "tap"
    w_other = ctx.width(other)

    def _solid(q):
        hits = ctx.plant.collisions_for({"cup": q, other: L.REST_Q}, {"cup": w0, other: w_other}, ctx.collision_scene())
        return [h for h in hits if "pad" not in "".join(h) and "finger" not in "".join(h)]

    leaving = {tuple(sorted(h)) for h in _solid(ctx.q("cup"))}
    cut = len(kept)
    for i, q in enumerate(kept):
        if any(tuple(sorted(h)) not in leaving for h in _solid(q)):
            cut = i
            break
    kept = kept[:cut]
    if len(kept) < 16 or float(kin.fk(kept[-1])[2, 0]) < 0.96:
        raise SkillFault("the rising roll is not reachable from this grasp "
                         f"({len(kept)} of {len(poses)} poses, cut {cut}, hand x {float(kin.fk(kept[-1])[2, 0]) if kept else 0:.2f}; {last_fail})")
    try:
        # Joint interpolation between the solved poses swings the hand ~30 mm and throws the cup.
        # Follow the Cartesian arc, seeded from the solution we already found.
        poses_ok = [kin.fk(q) for q in kept]
        traj = M.cart_move(kin, [ctx.tcp("cup")] + poses_ok, ctx.q("cup"), v_cart=0.16,
                           speed_scale=ctx.speed, min_T=1.6, max_joint_step=0.35, step_m=0.008,
                           timing="travel")
        ctx.check_traj("cup", traj, other_q=L.REST_Q, grip_width=w0)
    except PlanError as exc:
        raise SkillFault(f"the rising roll would hit something ({exc})")
    ctx.monitor("cup", True)
    note = {}

    def on_tick(k: int) -> None:
        if note or ctx.mon["cup"].grip_force >= 4.0:
            return
        # gripper effort and the hand pose only; the cup's true pose is the referee's
        note["s"] = f"grip {ctx.mon['cup'].grip_force:.2f} N at tcp z {float(ctx.tcp('cup')[2, 3]):.3f}"

    yield from ctx.play("cup", traj, on_tick=on_tick)
    slipped = ctx.mon["cup"].grip_force < 2.0 or ctx.width("cup") > w0 + 0.004
    grip = ctx.mon["cup"].grip_force
    ctx.monitor("cup", False)
    if slipped:
        steps = [float(np.max(np.abs(kept[i + 1] - kept[i]))) for i in range(len(kept) - 1)]
        worst = int(np.argmax(steps)) if steps else 0
        bow = 0.0
        for q in traj.q[::20]:
            p = kin.fk(q)[:3, 3]
            bow = max(bow, min(float(np.linalg.norm(p - kin.fk(w)[:3, 3])) for w in kept))
        raise SkillFault(f"the cup slipped in the jaws during the roll (grip {grip:.1f} N; {note.get('s', 'no drop seen')}; "
                         f"worst joint step {max(steps) if steps else 0:.2f} rad at {worst}, path bow {bow * 1000:.0f} mm)")
    x_up = float(ctx.tcp("cup")[2, 0])
    return {"msg": (f"wrist rolled {np.degrees(abs(float(ctx.q('cup')[5]) - j6_0)):.0f} deg while rising "
                    f"(started {plan['ao']:+d} deg, the other way); hand x is {x_up:.2f} up, cup flat")}


def s_flip_set(ctx: Ctx):
    """Lower the upright cup straight down until its base sits flat on the mat. No edge, no rock."""
    kin = ctx.kin["cup"]
    T0 = ctx.tcp("cup")
    # if the roll left the cup leaning, finish the untilt before the descent
    if float(T0[2, 0]) < 0.92 and ctx.scratch.get("flat_pose") is not None:
        goal = ctx.scratch["flat_pose"].copy()
        goal[:3, 3] = T0[:3, 3]
        try:
            fix = M.cart_move(kin, [T0, goal], ctx.q("cup"), v_cart=0.05, speed_scale=ctx.speed, min_T=0.8, max_joint_step=0.28)
            ctx.check_traj("cup", fix, grip_width=ctx.width("cup"))
            yield from ctx.play("cup", fix)
            T0 = ctx.tcp("cup")
        except PlanError:
            pass
    # A side hold on the base band cannot reach the mat: the wrist housing meets the table about
    # 2 cm above a seated base. Go to the lowest pose that stays clear and set the cup down there.
    z_floor = float(T0[2, 3])
    q_prev = ctx.q("cup")
    for z in np.arange(float(T0[2, 3]) - 0.008, L.MAT_SURFACE_Z + 0.010, -0.004):
        Tz = T0.copy()
        Tz[2, 3] = float(z)
        r = kin.ik_multi(Tz, [q_prev])
        if not r.ok:
            break
        hits = ctx.plant.collisions_for({"cup": r.q, "tap": ctx.q("tap")},
                                        {"cup": ctx.width("cup"), "tap": ctx.width("tap")}, ctx.collision_scene())
        if any("pad" not in "".join(h) and "finger" not in "".join(h) for h in hits):
            break
        z_floor = float(z)
        q_prev = r.q
    T1 = T0.copy()
    T1[2, 3] = z_floor
    try:
        traj = M.cart_move(kin, [T0, T1], ctx.q("cup"), v_cart=0.03, speed_scale=ctx.speed, min_T=1.2, max_joint_step=0.22)
    except PlanError as exc:
        raise SkillFault(f"cannot lower the flat cup onto the mat ({exc})")
    ctx.monitor("cup", True)
    st = {"n": 0, "k": 0}

    def on_tick(k: int) -> None:
        st["k"] = k

    def guard() -> bool:
        m = ctx.mon["cup"]
        if float(m.tcp[2, 3]) > 0.08:
            return False
        z_cmd = float(kin.fk(traj.q[min(st["k"], len(traj) - 1)])[2, 3])
        if float(m.delta_force(60)[2]) > 0.8 or float(m.tcp[2, 3]) - z_cmd > 0.004 or float(m.tcp[2, 3]) <= z_floor + 0.002:
            st["n"] += 1
        else:
            st["n"] = 0
        return st["n"] >= 4

    yield from ctx.play("cup", traj, guard=guard, on_tick=on_tick)
    ctx.monitor("cup", False)
    tcp = ctx.tcp("cup")
    yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M)
    yield from ctx.wait(0.9)
    ctx.bel.cup_xy = tcp[:2, 3].copy()
    T_up = ctx.tcp("cup")
    T_hi = T_up.copy()
    T_hi[2, 3] += 0.12
    try:
        up = M.cart_move(kin, [T_up, T_hi], ctx.q("cup"), v_cart=0.08, speed_scale=ctx.speed, min_T=0.7)
        ctx.check_traj("cup", up, grip_width=L.GRIPPER_MAX_OPEN_M)
        yield from ctx.play("cup", up)
    except PlanError:
        pass
    return {"msg": (f"cup released at ({tcp[0, 3]:.3f}, {tcp[1, 3]:.3f}), TCP z {tcp[2, 3] * 100:.1f} cm. "
                    f"The wrist housing cannot follow the base the last couple of centimetres, so the cup drops onto its base from there.")}
