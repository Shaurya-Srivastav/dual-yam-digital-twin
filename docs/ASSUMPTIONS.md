# Geometry and model assumptions

These are the numbers the simulator uses. They are not a new survey. The machine-readable list is `sim_station/assumptions.py`. A test checks that every name there is a real attribute of `sim_station.layout`.

| Name | Kind | Note |
|---|---|---|
| Cup rim radius | assumed | 76 mm. An earlier note said about 90 mm. Not remeasured. |
| Cup base, height, wall, masses | assumed | Nominal cup. Not remeasured. |
| Nest step, tether break, fatigue length | assumed | The nested-cup model is scripted. |
| Cup-arm mount, Level 1 waypoints | recorded | Already in the repository. |
| Tap-base offset | derived | From the two-grasp calculation, not a rail measurement. |
| Coaster dish radii | fitted | Widened so the recorded grasp, 2 cm off the placed axis, can sit in the dish. |
| Level 5 search box | assumed | A reachable patch of the mat, not the whole mat. |

`Plant.stack_assist` defaults to on. That is the synthetic upright torque. Setting it false is the ablation. The published nominal runs leave it on. Turning it off is not a claim about the real stack.
