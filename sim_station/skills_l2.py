"""Level 2 and 3 skills: take the top cup off an inverted nested stack and turn it upright with one hand.

The cup hand comes in with its wrist rolled half a turn the other way, so the inverted cup sits in it the way an
upright cup sits in the normal side grasp, only upside down. The tap hand pins the lower stack while the cup is
pulled free. Rolling the wrist back to its normal position then leaves the cup upright in the ordinary pour grip:
no second hand on the cup, and the cup never goes back down until the coaster.

Everything here uses sensing only: the grasp tells the stack position, the pin pads centre the stack, and the
force drop tells when the nested cups let go.
"""

from __future__ import annotations

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.kin import frame_from, rot_about
from sim_station.motion import PlanError
from sim_station.runtime import CTRL_STEPS, Ctx, SkillFault
from sim_station.sensing import Baseline
from sim_station.skills import HOLD_DZ, Z_UP, fmt_mm

PIN_BAND_BELOW_RIM = 0.0135       # tap pad centre below the top cup's rim (the funnel pad is 19 mm tall)


def top_face_z(n: int) -> float:
    return L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP + L.CUP_HEIGHT


def stack_rim_z(n: int) -> float:
    return L.MAT_SURFACE_Z + 0.0004 + (n - 1) * L.NEST_STEP


# ---------------------------------------------------------------------------
# cup arm: take hold of the top cup with the wrist rolled the other way
# ---------------------------------------------------------------------------

# A level hand-up grasp at the stack puts J5 past its stop, so the approach dips 8-12 deg. The roll back is about
# the approach, so the cup comes out leaning twice that; the carry to the spout takes the lean out (hold_pitch).
PREGRASP = ((-65, -8), (65, -8), (-65, -12), (65, -12), (-70, -8), (70, -8))
ROLL_Z = (0.275, 0.30, 0.24)


def _flip_frame(kin, xy, z, az_off_deg, tilt_deg):
    """The side grasp at an azimuth offset, rolled half a turn about the approach: hand x points up."""
    base = kin.mount[:2]
    radial = np.asarray(xy, float) - base
    radial = radial / np.linalg.norm(radial)
    ao = np.radians(az_off_deg)
    h = np.array([np.cos(ao) * radial[0] - np.sin(ao) * radial[1], np.sin(ao) * radial[0] + np.cos(ao) * radial[1], 0.0])
    t = np.radians(tilt_deg)
    a = np.array([np.cos(t) * h[0], np.cos(t) * h[1], np.sin(t)])
    j = np.cross([0.0, 0.0, 1.0], a)
    j /= np.linalg.norm(j)
    # The roll back runs J6 toward zero, which is the positive turn on the -azimuth side.
    return frame_from(a, -j, [xy[0], xy[1], z]), a, (1.0 if az_off_deg < 0 else -1.0)


def _flip_seeds(sgn: float):
    return [
        np.array([0.37 * sgn, 2.25, 1.67, -0.5, 1.41 * sgn, -2.07 * sgn]),
        np.array([0.0, 1.2, 1.1, 0.5, 0.0, -1.6 * sgn]),
        np.array([0.3, 1.0, 1.0, 0.8, 0.6, -1.6 * sgn]),
        np.array([0.1, 1.6, 1.2, 0.8, 0.0, -1.4 * sgn]),
        np.array([0.4, 1.8, 1.3, 0.6, -0.4, -1.5 * sgn]),
    ]


def flip_motion(kin, T0: np.ndarray, a: np.ndarray, sgn: float, xy, z_roll: float) -> list:
    """Rise clear of the stack, then roll the wrist back half a turn while the hand keeps moving.

    The roll only starts once the cup is off to the side of the stack: turning in the nest knocks it out of the fingers.
    """
    away = np.asarray(xy, float) - kin.mount[:2]
    away = away / max(float(np.linalg.norm(away)), 1e-6)
    R0 = T0[:3, :3].copy()
    p0 = T0[:3, 3].copy()
    z_clear = max(0.31, float(p0[2]))
    poses = []

    def add(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(p, float)
        poses.append(T)

    for z in np.linspace(float(p0[2]), z_clear, 4)[1:]:
        add(R0, [p0[0], p0[1], z])
    for s in (0.05, 0.09):
        add(R0, [p0[0] - away[0] * s, p0[1] - away[1] * s, z_clear])
    for z in np.linspace(z_clear, z_roll, 3)[1:]:
        add(R0, [p0[0] - away[0] * 0.09, p0[1] - away[1] * 0.09, z])
    for i, phi in enumerate(np.linspace(0.0, np.pi, 14)[1:], start=1):
        s = 0.09 + 0.025 * i / 13.0
        add(rot_about(a, sgn * phi) @ R0, [p0[0] - away[0] * s, p0[1] - away[1] * s, z_roll])
    return poses


def plan_pregrasp(ctx: Ctx) -> dict:
    """Pick the hand-up attitude whose approach, grasp and roll back are all reachable and collision-free."""
    n = ctx.bel.stack_n
    if n is None:
        raise SkillFault("stack height unknown: find it with the peck first")
    kin = ctx.kin["cup"]
    xy = np.asarray(ctx.bel.stack_xy, float)
    # The middle of the inverted cup is the band the pour grip holds once the cup is upright.
    z = top_face_z(n) - HOLD_DZ
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
        rg = kin.ik_multi(T1, ([rp.q] if rp.ok else []) + seeds)
        if not (rp.ok and rg.ok):
            last = f"{ao:+d}/{tilt}: " + (rp.reason or rg.reason or "grasp")[:80]
            continue
        for z_roll in ROLL_Z:
            qs, why = _chain(kin, flip_motion(kin, T1, a, sgn, xy, z_roll), seeds, rg.q)
            if qs is None:
                last = f"{ao:+d}/{tilt} roll at {z_roll:.3f}: {why[:80]}"
                continue
            try:
                app_q = [q_now] + ([rh.q] if rh.ok else []) + [rp.q]
                app = M.via_move(app_q, speed_scale=ctx.speed, min_T=1.0)
                ctx.check_traj("cup", app, grip_width=L.GRIPPER_MAX_OPEN_M)
                flip = M.via_move(qs, speed_scale=ctx.speed, min_T=3.2)
                ctx.check_traj("cup", flip, other_q=L.REST_Q, grip_width=L.CUP_HOLD_WIDTH_ON_AXIS)
            except PlanError as exc:
                last = f"{ao:+d}/{tilt} roll at {z_roll:.3f}: {str(exc)[:100]}"
                continue
            return dict(ao=ao, tilt=tilt, sgn=sgn, a=a.copy(), grasp=T1, pre=T_pre, z_roll=z_roll,
                        q_hi=rh.q if rh.ok else None, q_pre=rp.q, z=z, n=n)
    raise SkillFault(f"no hand-up grasp whose roll back is reachable ({last})")


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


def s_approach_stack(ctx: Ctx):
    plan = plan_pregrasp(ctx)
    ctx.scratch["pregrasp"] = plan
    ctx.scratch["stack_fr"] = dict(grasp=plan["grasp"], a=plan["a"], pre=plan["pre"])
    qs = [ctx.q("cup")] + ([plan["q_hi"]] if plan.get("q_hi") is not None else []) + [plan["q_pre"]]
    traj = M.via_move(qs, speed_scale=ctx.speed, min_T=1.0)
    ctx.check_traj("cup", traj, grip_width=L.GRIPPER_MAX_OPEN_M)
    yield from ctx.play("cup", traj)
    return {"msg": (f"wrist rolled half a turn the other way (hand x up), approach {plan['ao']:+d} deg off radial and "
                    f"{abs(plan['tilt']):.0f} deg down; stack of {plan['n']}, TCP {plan['z'] * 100:.1f} cm on the middle band")}


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
    return {"msg": "fingers straddle the top cup's middle band"}


def s_close_stack(ctx: Ctx):
    """Close on the top cup. A short chord means the hand is off the axis; step along the approach and close again."""
    cap = L.FORCE_CAP_STACK_TOP_N
    fr = ctx.scratch["stack_fr"]
    kin = ctx.kin["cup"]
    a = fr["a"]
    r_c = 0.5 * L.CUP_HOLD_WIDTH_ON_AXIS
    sign, s_prev, deep = 1.0, None, 0.0
    last_w = 0.0
    for n in range(6):
        res = yield from ctx.grip_close("cup", cap, rate=0.04)
        w = float(res["width"])
        last_w = w
        s = None
        if res["stalled"] and w < 2.0 * r_c:
            s = float(np.sqrt(max(r_c * r_c - (0.5 * w) ** 2, 0.0)))
        # A stall inside the old band is the grasp the pour already uses. Only a shorter chord is off the axis.
        if res["stalled"] and 0.058 < w < 0.074:
            ctx.bel.found["stack_tcp"] = ctx.tcp("cup")[:3, 3].copy()
            ctx.bel.cup_in_hand = True
            ctx.scratch["top_hold_width"] = w
            extra = f" after {n} re-try(ies)" if n else ""
            return {"msg": f"jaws stalled at {fmt_mm(w)} with {res['force']:.1f} N of {cap:.0f} N: top cup gripped{extra}"}
        yield from ctx.grip_open("cup", L.GRIPPER_MAX_OPEN_M)
        if s is not None and s > 0.004:
            if s_prev is not None and s > s_prev + 0.002:
                sign = -sign
            step = sign * min(s, 0.016)
            s_prev = s
            deep += step
            ctx.event(f"stalled on a {w * 1000:.0f} mm chord, about {s * 1000:.0f} mm off the axis; moving {step * 1000:+.0f} mm")
        else:
            deep += 0.010
        T = fr["grasp"].copy()
        T[:3, 3] = fr["grasp"][:3, 3] + a * deep
        try:
            move = M.cart_move(kin, [ctx.tcp("cup"), T], ctx.q("cup"), v_cart=0.04, speed_scale=ctx.speed, min_T=0.45)
            ctx.check_traj("cup", move, grip_width=L.GRIPPER_MAX_OPEN_M)
            yield from ctx.play("cup", move)
        except PlanError:
            deep = -0.5 * deep
    raise SkillFault(f"jaws stalled at {fmt_mm(last_w)}; the top cup's middle band is about 67 mm")


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
    tilts = ((0.0, -20.0, -35.0, -50.0, -65.0, -75.0) if z_pin >= 0.05
             else (-25.0, -40.0, -55.0, -65.0, -75.0, -15.0, 0.0))
    last = ""
    for az_deg in (60.0, 45.0, 75.0, 90.0, 30.0, 15.0, 0.0, -15.0, -30.0, -45.0, 105.0, 120.0, 135.0,
                   150.0, 165.0, 180.0, -60.0, -75.0, -90.0, -120.0, -150.0):
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
    # The stack stands on the mat, so its axis is vertical whatever the hand's dip.
    pull = Z_UP.copy()
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
# cup arm: roll the wrist back
# ---------------------------------------------------------------------------

def s_flip_lift(ctx: Ctx):
    """Roll the wrist back to its normal position. The cup comes out upright in the ordinary pour grip."""
    plan = ctx.scratch.get("pregrasp")
    if plan is None:
        raise SkillFault("the wrist was not rolled before the grasp")
    kin = ctx.kin["cup"]
    w0 = ctx.width("cup")
    # Tracking during the pull leaves the wrist a few degrees off the solved attitude; roll from the solved one.
    T0 = ctx.tcp("cup").copy()
    T0[:3, :3] = plan["grasp"][:3, :3]
    T0[:2, 3] = plan["grasp"][:2, 3]
    poses = flip_motion(kin, T0, plan["a"], plan["sgn"], ctx.bel.stack_xy, plan["z_roll"])
    seeds = _flip_seeds(plan["sgn"])
    qs, qprev, missed, why = [], ctx.q("cup"), 0, ""
    for i, Tp in enumerate(poses):
        r = kin.ik_multi(Tp, [qprev] + seeds)
        if not r.ok or (qs and float(np.max(np.abs(r.q - qs[-1]))) > 0.55):
            missed += 1
            why = f"pose {i}: " + (r.reason if not r.ok else "joint jump")
            if missed > 2 or i == len(poses) - 1:
                raise SkillFault(f"the roll back is not reachable from this grasp ({why})")
            continue
        missed = 0
        qs.append(r.q)
        qprev = r.q
    try:
        # Joint interpolation between the solved poses swings the hand and throws the cup; follow the Cartesian arc.
        traj = M.cart_move(kin, [ctx.tcp("cup")] + [kin.fk(q) for q in qs], ctx.q("cup"), v_cart=0.16,
                           speed_scale=ctx.speed, min_T=1.6, max_joint_step=0.35, step_m=0.008, timing="travel")
        ctx.check_traj("cup", traj, grip_width=w0)
    except PlanError as exc:
        raise SkillFault(f"the roll back would hit something ({exc})")
    ctx.monitor("cup", True)
    yield from ctx.play("cup", traj)
    grip = ctx.mon["cup"].grip_force
    ctx.monitor("cup", False)
    if grip < 2.0 or ctx.width("cup") > w0 + 0.004:
        raise SkillFault(f"the cup slipped in the jaws during the roll back (grip {grip:.1f} N, jaws {fmt_mm(ctx.width('cup'))})")
    # Cup axis (base to rim) in the hand frame, from the grasp: it pointed straight down then.
    c_h = plan["grasp"][:3, :3].T @ -Z_UP
    lean = float(np.degrees(np.arccos(np.clip(float((ctx.tcp("cup")[:3, :3] @ c_h)[2]), -1.0, 1.0))))
    # Approach pitch at which this hold leaves the cup vertical; every spout and coaster frame adds it.
    ctx.scratch["hold_pitch"] = float(np.degrees(np.arcsin(np.clip(c_h[2], -1.0, 1.0))))
    ctx.bel.cup_in_hand = True
    return {"msg": (f"wrist rolled back half a turn to its normal position (hand x {float(ctx.tcp('cup')[2, 0]):+.2f}): "
                    f"cup upright in the pour grip, leaning {lean:.0f} deg until the carry pitches the hand "
                    f"{ctx.scratch['hold_pitch']:+.0f} deg")}
