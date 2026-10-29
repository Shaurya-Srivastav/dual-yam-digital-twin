# Status (2026-10-28, commit under this file)

Simulation only. Nothing has run on the arms.

| Claim | State |
|---|---|
| Nominal Levels 1–5, no randomization | Passing the strict referee on the selftest, as of the last run recorded in `docs/RESULTS.md`. Re-run `python -m sim_station.selftest` after any physics or skill change. |
| Randomized, 24 trials | 79 / 42 / 42 / 21 / 12 % against targets 98 / 98 / 95 / 95 / 90 %. Not at target. |
| Pose-aligned Level 1 replay | The cup is moved to the recorded grasp before playback. A pass is not independent proof of the station layout. |
| Scene-as-reset replay | Same waypoints, cup left where the level reset put it. Latest selftest: fail, 105.7 mm from the outlet axis and 19.7 mm from the coaster centre. Printed, not part of the exit code. |
| FK/IK check | This model's forward kinematics against its own inverse kinematics. Not a surveyed hand pose. |
| Hardware | `SimYam` is an adapter. The skills call `Plant` directly. Not a hardware port. |
| Geometry | Rim diameter in `layout.py` is 76 mm. An earlier onsite estimate was about 90 mm. Not remeasured. Coaster dish slack, tap-base offset and spigot shape are inferred. Setup photos are not in this tree. |
| Stack model | A restoring torque and a scripted nested-cup tether are still in the plant. They are not calibrated from a pull test. |
| Planner | Arm paths are checked against the bench, the dispenser, the other arm, and against the cup, remaining stack, coaster and lever placed at the belief. Fingertip pads may touch a cup, the stack and the lever. The live object pose is not copied into that query. Until Level 3 has counted the cups, the remaining stack's vertical shape is still the simulator's enabled geoms. |

`python -m sim_station.selftest` exits 0 only when the five nominal levels, the pose-aligned replay and the FK/IK check pass. A green selftest does not update the randomized table.

Public repo: no song files. Dance clips on the project page are silent. Local playback uses files named in `sim_station/audio.py`.
