# Abstract

This is a MuJoCo simulation of a two-arm cup station. One arm picks a paper or plastic cup, holds it under a dispenser, and sets it in a coaster. The other arm lifts a lever. The pour is a timer: the model contains no water. The skills do not see cameras or the simulator's object poses. They read joint position, velocity, motor torque and gripper width, and they infer contact from the torque residual. A separate referee scores the true poses after the run.

On the nominal seed, all five levels meet the strict checks: the cup ends upright and within 15 mm of the coaster, the lever is open for 7.0 ± 0.2 s, and the forbidden contacts stay clear. On 24 randomized trials scored 2026-10-29, Levels 1–3 passed 96, 79 and 67 percent, and Level 5 passed 24 of 24. Level 4 was not rescored on that code. The stack upright torque and the nested-cup tether are synthetic. The 76 mm rim is the model's assumption. No motion in this repository has been run on the physical arms.
