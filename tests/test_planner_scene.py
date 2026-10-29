"""Planner collision queries use the belief, not the live object pose.

    python tests/test_planner_scene.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from sim_station.runner import Session


def _arm_over(S, xy, z=0.12):
    """A cup-arm joint vector whose fingertip is near xy, if IK can reach it. Returns q or None."""
    ctx = S.ctx
    T = ctx.tcp("cup").copy()
    T[:3, 3] = [float(xy[0]), float(xy[1]), z]
    r = ctx.kin["cup"].ik(T, ctx.q("cup"))
    return r.q if r.ok else None


def test_cup_query_follows_belief_not_truth():
    S = Session()
    S.reset(1, "paper", seed=0)
    P = S.plant
    true = P.cup_state()["xyz"][:2].copy()
    moved = true + np.array([0.12, 0.0])
    qa = P.free_qadr[P.active_cup]
    P.data.qpos[qa: qa + 2] = moved
    import mujoco
    mujoco.mj_forward(P.model, P.data)
    assert np.linalg.norm(P.cup_state()["xyz"][:2] - true) > 0.05
    q = S.ctx.q("cup")
    scene_at_belief = dict(cup_xy=true, coaster_xy=S.ctx.bel.coaster_xy, lever_yaw=float(S.ctx.bel.lever_yaw), cup_in_hand=False)
    scene_at_truth = dict(cup_xy=moved, coaster_xy=S.ctx.bel.coaster_xy, lever_yaw=float(S.ctx.bel.lever_yaw), cup_in_hand=False)
    # The same arm pose. Hits may or may not include the cup; the point is they must not depend on
    # where the live body was moved, only on the xy passed in.
    h_bel = P.collisions_for({"cup": q, "tap": S.ctx.q("tap")}, scene=scene_at_belief)
    # Put the live cup somewhere else again and repeat the belief query.
    P.data.qpos[qa: qa + 2] = true + np.array([-0.2, 0.15])
    mujoco.mj_forward(P.model, P.data)
    h_bel_2 = P.collisions_for({"cup": q, "tap": S.ctx.q("tap")}, scene=scene_at_belief)
    assert h_bel == h_bel_2, (h_bel, h_bel_2)
    # And a query that names the moved xy must be allowed to differ from the belief query.
    h_far = P.collisions_for({"cup": q, "tap": S.ctx.q("tap")}, scene=scene_at_truth)
    assert h_bel == h_bel_2
    _ = h_far


def test_unknown_lever_is_not_read_from_the_plant():
    S = Session()
    S.reset(4, "paper", seed=0)
    P = S.plant
    P.data.qpos[P.yaw_q] = 2.2
    import mujoco
    mujoco.mj_forward(P.model, P.data)
    q = {a: S.ctx.q(a) for a in ("cup", "tap")}
    hidden = P.collisions_for(q, scene=dict(lever_yaw=None, cup_in_hand=False, coaster_xy=S.ctx.bel.coaster_xy))
    named = P.collisions_for(q, scene=dict(lever_yaw=2.2, cup_in_hand=False, coaster_xy=S.ctx.bel.coaster_xy))
    # With yaw unknown the lever geoms are omitted, so the result must not change if the live yaw changes.
    P.data.qpos[P.yaw_q] = -2.2
    mujoco.mj_forward(P.model, P.data)
    hidden_2 = P.collisions_for(q, scene=dict(lever_yaw=None, cup_in_hand=False, coaster_xy=S.ctx.bel.coaster_xy))
    assert hidden == hidden_2
    _ = named


def test_level4_belief_serializes():
    S = Session()
    S.reset(4, "paper", seed=0)
    yaw = float(S.ctx.bel.lever_yaw)
    assert not np.isfinite(yaw)
    payload = dict(
        lever_yaw=None if not np.isfinite(yaw) else round(float(np.degrees(yaw)), 1),
        lever_yaw_known=bool(np.isfinite(yaw)),
    )
    raw = json.dumps(payload, allow_nan=False)
    assert json.loads(raw)["lever_yaw"] is None


def test_replay_labels():
    from sim_station import proofs
    import inspect
    src = inspect.getsource(proofs.replay_l1)
    assert "align_cup" in src
    # the aligned path writes the grasp XY; the scene path must be able to skip it
    assert "if align_cup" in src


def main() -> int:
    tests = [test_cup_query_follows_belief_not_truth, test_unknown_lever_is_not_read_from_the_plant,
             test_level4_belief_serializes, test_replay_labels]
    bad = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
