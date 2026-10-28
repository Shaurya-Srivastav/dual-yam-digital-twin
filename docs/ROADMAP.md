# Roadmap: what is still open

Ordered by what blocks the most.

1. **Stack on the coaster spot (Levels 1-4 all start at the coaster).** Needs the tap arm's base at the measured solution (cup-frame (0.067, -0.480), no relative yaw). Tried: the lever lift then fails IK because the lever arc was tuned for the current base. Work: re-derive the lever grip frames for the new base, widen the lift arc's IK seeds, rerun the Level 1/4 Monte Carlo, then move the L2/L3 stack and put the L2 serve spot elsewhere.
2. **Stack contact in Levels 2/3.** Arm links and housing brush the remaining stack on the way to the spout (referee: `arm_stack`). With the stack on the coaster spot this should disappear; until then `Plant.stack_is_obstacle = True` makes the planner refuse such paths (no route exists in the current layout).
3. **Randomized success rate.** 79 / 38 / 21 / 25 / 33 % against targets 98 / 98 / 95 / 95 / 90 %. Known causes: top cup slipping in the pull, some stack heights without a reachable side grasp, the held cup bumping the valve, the wide coaster dish not centring the cup.
4. **Level 5 speed and contacts.** Information-gain pecking (next peck chosen by expected entropy reduction on the cup-position belief), pecks planned so the housing never reaches a feasible cup position.
5. **Calibration from the station photos.** Tap base spacing, coaster dish radius, spigot geometry are inferred, not measured.
6. **Hardware.** Run `python -m sim_station.sim_robot --hardware` at half speed and fit friction and gravity factors against the twin's tracking error.
7. **Dance.** Tune the lead (`dance.LEAD_S`) against real speakers; add more moves per style.
