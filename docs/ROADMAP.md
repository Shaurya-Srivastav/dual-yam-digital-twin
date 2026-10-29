# Roadmap: what is still open

Ordered by what blocks the most. Rates are from [RESULTS.md](RESULTS.md).

1. **Cup centring after the flip (Levels 2 and 3).** The coaster is bolted. A cup in it may start ±2 mm off centre, and the dish does not pull a seated cup back inside 15 mm. Three seeds land past that line because the cup walks along the jaw during the carry and the jaw opening cams it farther. Pressing, sliding, and lowering before the open were tried and did not bring those seeds inside 15 mm.
2. **Level 4, not rescored.** Handles near the pour angle can be squared at a 15 mm stall and parked with the dispenser under 2 mm in single-yaw checks. Handles on the far side of the tank are felt, and the short push that should turn them meets the tank or runs out of time. The old 8 % table used a fingertip sweep this tree no longer runs. A 24-trial on the current search has not been run.
3. **Level 3 pin reachability.** No collision-free pin approach for three of 24 stack heights (seeds 17, 20, 21).
4. **Level 5, not rescored on this pour.** The search is a flat-pad stomp, cup arm on +y and tap arm on −y. Earlier on 2026-10-29 that search passed 23 of 24, with one pad hitting the tank. The pour path changed after that run.
5. **Grasp misses that remain.** Level 1 seed 22 closes on nothing at the handle. Level 2 and Level 3 seed 22 stall short of the top cup's middle band.
6. **Not separate implementations yet:** torque actuators (the arms are position actuators with the MIT law in the command), bimanual zone locks, a `GripperForceLimiter` class, a particle-filter or expected-entropy Level 5 belief.
7. **Calibration from the station photos.** The photos are not in the tree; tap base spacing, dish radius and spigot geometry are inferred, not measured.
8. **Hardware.** Run `python -m sim_station.sim_robot --hardware` at half speed and fit friction and gravity factors against the twin's tracking error.
9. **Dance.** Tune the lead (`dance.LEAD_S`) against real speakers; add more moves per style. Take On Me is tracked at 84 BPM and danced on every beat (not half time).
