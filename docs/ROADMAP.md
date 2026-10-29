# Roadmap: what is still open

Ordered by what blocks the most. Rates are from [RESULTS.md](RESULTS.md).

1. **Coaster centring (Levels 1-4).** The coaster is jittered up to 8 mm in randomized trials and the skills only know its nominal spot. The dish leaves about 2 cm of slack around the cup base (needed by the real replay), so it does not centre the cup. Next: localise the coaster by touch before the serve (a peck on its rim, not a slide of the held cup, which was tried and made it worse), or measure the real dish and tighten it.
2. **Level 4 dispenser creep and lever lift.** Parking the handle pushes the dispenser past 2 mm on 7 of 24 draws; a stiffer tape model fixed that but broke the Level 1 close timing. The lever lift's first IK step jumps 0.13 rad on 3 of 24. Next: a gentler park push, and a seeded first IK step for the lift.
3. **Level 3 pin reachability.** No collision-free pin approach for some stack heights (5 of 24). Next: more pin approach azimuths and heights.
4. **Level 5 scan.** A peck into the open mouth reads as empty mat. Next: model the mouth in the scan's candidate update without dropping hits elsewhere (the first attempt lowered the rate).
5. **Grasp retries.** 1-2 of 24 per level close on nothing or stall on a 61-63 mm chord after nine attempts.
6. **Not separate implementations yet:** torque actuators (the arms are position actuators with the MIT law in the command), bimanual zone locks, a `GripperForceLimiter` class, a particle-filter or expected-entropy Level 5 belief.
7. **Calibration from the station photos.** The photos are not in the tree; tap base spacing, dish radius and spigot geometry are inferred, not measured.
8. **Hardware.** Run `python -m sim_station.sim_robot --hardware` at half speed and fit friction and gravity factors against the twin's tracking error.
9. **Dance.** Tune the lead (`dance.LEAD_S`) against real speakers; add more moves per style. Take On Me is tracked at 84 BPM and danced on every beat (not half time).
