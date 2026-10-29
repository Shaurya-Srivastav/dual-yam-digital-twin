# Engineering roadmap

Source audit dated 2026-10-28, written against `main` at `09040e7`. Reconciled against the tree that contains this file before any edit from that audit. The audit could not execute MuJoCo. Findings below are marked confirmed-in-source or already-changed.

This is the work order. Later phases (imitation learning, reinforcement learning, hardware motion) do not start until the gates in front of them pass. No physical robot motion from this roadmap.

## What already changed after 09040e7

- The remaining stack is on the coaster spot and `stack_is_obstacle` defaults true. Nominal Levels 2 and 3 pass `stack_never_touched` in the last selftest. The audit's "L2/L3/L5 fail nominal" is stale.
- The Level 2 wrist is turned to the flip attitude before the grasp.
- Randomized rates were rerun after the one-arm flip and the lever-search change: 79 / 54 / 54 / 8 / 17 % (24 trials). Earlier runs: 79 / 42 / 42 / 21 / 12 % and 79 / 38 / 21 / 25 / 33 %.
- The public site states those rates and does not claim the targets are met.

## Confirmed, still open after that reconcile

- A01. `collisions_for` used to ignore every cup and the coaster, and it copied the live state. Addressed in the planner-scene change: the query places the cup, stack, coaster and lever from the belief argument. Pads may touch those objects. Unknown lever yaw is omitted rather than read from the plant. Remaining: until Level 3 sets `stack_n`, the stack's height is still the enabled geoms in the model. Decorative side stacks are not in the query.
- A02. `_stabilize_stacks` still applies an upright restoring torque. Not removed. Removing it without a measured support model would change every stack result. Needs an ablation, not a silent delete.
- A03. Nested cups still use the scripted tether. Not replaced with a measured pull curve.
- A04. `replay_l1` still moves the cup to the grasp XY when `align_cup` is true. It is now labeled `pose_aligned`. `replay_l1_scene` does not move the cup.
- A05. `equivalence_l1` is labeled FK/IK self-consistency.
- A06. Skills still call `Plant`. No new hardware path.
- A07–A08. No hardware run, no change to motor timeouts.
- A09. 76 mm rim vs an earlier ~90 mm estimate. Recorded in `layout.py` and `STATUS.md`. Number not changed.
- A11. "1:1" and "proof" wording removed from the project page, the dashboard and `proofs.py`.
- A12. Selftest matches the nominal passes. Its docstring says exit 0 is not a randomized result. The scene-as-reset replay is printed and is not in the exit code.
- A13. `docs/RESULTS.md` matches the rerun. It is still a hand-written table, not generated from JSON.
- A43. Level 4 belief yaw is `NaN` internally. The dashboard snapshot and the belief overlay now emit `null` and skip the marker instead of serializing `NaN`.

## Do not start yet

Behavior cloning, residual RL, active-sensing training, and any `get_yam_robot` motion. Dance Mode stays a local optional. The beat-tracker branch that was `if cand[k] > 0 or True` now takes the best finite predecessor. `tests/test_numerics.py` covers a click track and silence.

## Exit for the next change

A nominal selftest that still passes, `python tests/test_planner_scene.py` passing, and a sentence in `STATUS.md` if a nominal level starts failing the strict referee because the planner can now see the belief cup. Do not loosen the referee to get the exit code back.
