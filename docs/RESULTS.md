# Results (measured, simulation only)

**Real-data replay of the Level 1 Pi waypoints** (`python -m sim_station.proofs`): cup lifted from the coaster, **0.9 mm** from the outlet axis under the spout, back in the coaster 8.9 mm from centre, max tilt 6.9°. To get this the coaster dish had to be widened (the real grasp sits 2.0 cm from the real placement). Confirm the dish radius with calipers.

**IK vs the 12 real waypoints:** all reproduced within 0.4 mm and 0.01°.

**Nominal run of each level** (`python -m sim_station.selftest`, no randomization): Levels 1 and 4 pass the full referee; Levels 2 and 3 finish the pour and the serve (flow 6.98 s, cup 11-12 mm from centre) but the housing or forearm brushes the remaining stack; Level 5 finishes but the hand housing touches the target cup while pecking.

**Monte Carlo, 24 randomized trials per level** (positions ±1 cm, friction/mass ±25-30 %, gains ±10 %, base ±2 mm / 0.5°, noise, 2-5 tick delay), before the last fixes to Level 5:

| Level | Pass | Target |
|---|---|---|
| 1 | 79 % | 98 % |
| 2 | 38 % | 98 % |
| 3 | 21 % | 95 % |
| 4 | 25 % | 95 % |
| 5 | 33 % | 90 % |

**Known problems.**
- The remaining stack sits 11 cm from the line between the cup arm's base and the spout, so any carry to the spout brushes it. With the stack as a planning obstacle no route exists (`Plant.stack_is_obstacle`). The real layout puts the stack at the coaster spot with the tap base nearer to the cup arm (solution 1 in the layout notes); adopting it should clear the corridor but needs the tap base measured.
- Top cup slips during the pull on some randomized draws; some stack heights have no reachable side grasp; the held cup can bump the valve; the dish no longer centres the cup.
- Level 5 takes about 100 s of simulated time.

**Render:** 44 fps delivered at a 1400×900 viewport while simulating.
