# Results (measured, simulation only)

**Real-data replay of the Level 1 Pi waypoints** (`python -m sim_station.proofs`): cup lifted from the coaster, **0.9 mm** from the outlet axis under the spout, back in the coaster 8.9 mm from centre, max tilt 6.9°. To get this the coaster dish had to be widened (the real grasp sits 2.0 cm from the real placement). Confirm the dish radius with calipers.

**IK vs the 12 real waypoints:** all reproduced within 0.4 mm and 0.01°.

**Nominal run of each level** (`python -m sim_station.selftest`, no randomization): all five pass the full referee.

| Level | Cup from coaster centre | Flow | Wall time |
|---|---|---|---|
| 1 | 4 mm | 6.96 s | 12 s |
| 2 | 12 mm | 6.95 s | 58 s |
| 3 | 11 mm | 6.94 s | 63 s |
| 4 | 3 mm | 6.95 s | 13 s |
| 5 | 13 mm | 6.96 s | 22 s |

In Levels 2 and 3 the stack stands on the coaster spot (as on the real station) and the referee's `stack_never_touched` and `arm_stack` checks pass.

**Monte Carlo, 24 randomized trials per level** (positions ±1 cm, coaster ±8 mm unknown to the skills, friction/mass ±25-30 %, gains ±10 %, base ±2 mm / 0.5°, noise, 2-5 tick delay, every fourth trial plastic, Level 3 stacks of 3 to 11 cups), on the committed code:

| Level | Pass | Target | Main failures |
|---|---|---|---|
| 1 | 79 % | 98 % | cup >15 mm from centre (2), grasp (2), flow (1) |
| 2 | 42 % | 98 % | cup >15 mm from centre (9), collisions (2), cup not upright (2) |
| 3 | 42 % | 95 % | no pin approach (5), cup >15 mm (4), cup not upright (2) |
| 4 | 21 % | 95 % | dispenser moved >2 mm (7), flow (4), cup >15 mm (3), lever-lift IK jump (3) |
| 5 | 12 % | 90 % | scan found nothing (8), arms touched (3), grasp closed on nothing (3), parked stack touched (3), cup >15 mm (2) |

**Tried and taken out because they measured worse:**
- Feeling the dish wall on four sides with the held cup before release (cup ended 19-24 mm off).
- A stiffer taped dispenser (fixed the Level 4 creep, but the Level 1 lever then stayed open 25 s).
- An offset second Level 5 scan pass and model-based candidate pruning (Level 5 fell to 8 %).

**Render:** 44 fps delivered at a 1400×900 viewport while simulating.
