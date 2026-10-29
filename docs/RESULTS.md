# Results (measured, simulation only)

**Real-data replay of the Level 1 Pi waypoints** (`python -m sim_station.proofs`): cup lifted from the coaster, **0.9 mm** from the outlet axis under the spout, back in the coaster 8.9 mm from centre, max tilt 6.9°. To get this the coaster dish had to be widened (the real grasp sits 2.0 cm from the real placement). Confirm the dish radius with calipers.

**Scene-as-reset replay** (same waypoints, cup not moved onto the grasp): fail. 105.7 mm from the outlet axis, 19.7 mm from the coaster centre. The aligned replay above is not this test.

**IK vs the 12 real waypoints:** this model's forward kinematics, reproduced within 0.4 mm and 0.01°. Not a surveyed hand pose.

**Nominal run of each level** (no randomization, the runs rendered to `site/videos/`): all five pass the full referee.

| Level | Cup from coaster centre | Flow | Simulated time |
|---|---|---|---|
| 1 | 3.4 mm | 6.95 s | 55 s |
| 2 | 4.9 mm | 6.95 s | 102 s |
| 3 | 4.5 mm | 6.95 s | 109 s |
| 4 | 4.1 mm | 6.95 s | 70 s |
| 5 | 13.8 mm | 6.94 s | 76 s |

In Levels 2 and 3 one arm takes the top cup off the stack with its wrist rolled half a turn, then rolls back so the cup is upright in the pour grip. Nothing is handed over or set down. The stack stands on the coaster spot (as on the real station) and the referee's `stack_never_touched` and `arm_stack` checks pass.

**Monte Carlo, 24 randomized trials** (`python -m sim_station.evaluate --trials 24 --seed0 0`; positions ±1 cm, coaster bolted, a cup in the coaster ±2 mm, stacks ±8 mm, friction/mass ±25-30 %, gains ±10 %, base ±2 mm / 0.5°, noise, 2-5 tick delay, every fourth trial plastic, Level 3 stacks of 3 to 11 cups). Levels 1–3 and 5 below were scored on 2026-10-29. Level 4 was not.

| Level | Pass | Wilson 95 % | Target | Previous run | Main failures |
|---|---|---|---|---|---|
| 1 | 23/24, 96 % | 80-99 % | 98 % | 79 % | handle pinch closed on nothing (seed 22) |
| 2 | 19/24, 79 % | 60-91 % | 98 % | 54 % | cup >15 mm from the bolted coaster (seeds 0, 1, 12), short grasp (seed 22), cup tilted (seed 17) |
| 3 | 16/24, 67 % | 47-82 % | 95 % | 54 % | cup >15 mm (seeds 0, 1, 12), no pin pose (seeds 17, 20, 21), pour timing (seed 5), short grasp (seed 22) |
| 4 | not rerun | — | 95 % | 8 % on 2026-10-28 | that 8 % used a fingertip sweep this tree no longer runs |
| 5 | 24/24, 100 % | 86-100 % | 90 % | 23/24 earlier the same day | none. The earlier miss was a handle pinch that closed on nothing; a second close around the paddle clears it. Pass-only mean 180 s, peak grip 29 N, max tilt 25°. |

The coaster does not move. The three Level 2 and Level 3 misses are the cup walking in the fingers during the carry, then the jaw opening camming it past 15 mm. The dish funnel does not pull it back. Level 1's nominal pour on this code is 6.95 s with the dispenser moved 1.38 mm. One Level 2 trial in an earlier table logged a multi-kilonewton grip spike; the 2026-10-29 Level 2 run's peak was a single-step contact glitch of the same kind, not a sustained squeeze.

**Tried and taken out because they measured worse:**
- Feeling the dish wall on four sides with the held cup before release (cup ended 19-24 mm off).
- A stiffer taped dispenser (fixed the Level 4 creep, but the Level 1 lever then stayed open 25 s).
- An offset second Level 5 scan pass and model-based candidate pruning (Level 5 fell to 8 %).

**Render:** 44 fps delivered at a 1400×900 viewport while simulating.
