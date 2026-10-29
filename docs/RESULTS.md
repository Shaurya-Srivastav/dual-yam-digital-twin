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

**Monte Carlo, 24 randomized trials per level** (`python -m sim_station.evaluate --trials 24 --seed0 0`; positions ±1 cm, coaster ±8 mm unknown to the skills, friction/mass ±25-30 %, gains ±10 %, base ±2 mm / 0.5°, noise, 2-5 tick delay, every fourth trial plastic, Level 3 stacks of 3 to 11 cups), on the code in this commit:

| Level | Pass | Wilson 95 % | Target | Previous run | Main failures |
|---|---|---|---|---|---|
| 1 | 19/24, 79 % | 60-91 % | 98 % | 79 % | grasp missed (2), cup >15 mm from centre (2), forbidden contact (1) |
| 2 | 13/24, 54 % | 35-72 % | 98 % | 42 % | cup >15 mm (5), pour timing (2), lever IK jump, no pin pose, forbidden contact, tilt (1 each) |
| 3 | 13/24, 54 % | 35-72 % | 95 % | 42 % | no pin pose (5), cup >15 mm (4), forbidden contact (1), pour timing (1) |
| 4 | 2/24, 8 % | 2-26 % | 95 % | 21 % | pour timing (9), dispenser moved >2 mm (7), cup >15 mm (2), grasp missed (2) |
| 5 | 4/24, 17 % | 7-36 % | 90 % | 12 % | grasp closed on nothing (12), arms touched (4), fingers on the rim (3) |

Levels 2 and 3 improved with the one-arm flip. Level 4 fell: the new lever search reaches only the radii on the open side of the tank, so when the handle lies behind it a fingertip sweep pushes the handle round, and that drags the taped dispenser. One Level 2 trial (seed 19) logged a 4684 N peak grip force, a single-step contact spike, not a sustained squeeze.

**Tried and taken out because they measured worse:**
- Feeling the dish wall on four sides with the held cup before release (cup ended 19-24 mm off).
- A stiffer taped dispenser (fixed the Level 4 creep, but the Level 1 lever then stayed open 25 s).
- An offset second Level 5 scan pass and model-based candidate pruning (Level 5 fell to 8 %).

**Render:** 44 fps delivered at a 1400×900 viewport while simulating.
