# Questions

**Why are there no cameras?** The skills are written to use the same signals a motor driver reports: joint position, velocity, torque, gripper width and gripper effort. Object pose is hidden from them on purpose.

**How is contact detected?** The monitor subtracts a gravity term and a friction term from the motor torque and maps the residual through the Jacobian. It is not a load cell, and it is not the simulator's contact force.

**How is a run scored?** `runner.check_pass` reads the true poses after the motion. The cup must be within 10° of upright and 15 mm of the coaster centre. The lever must be open 7.0 ± 0.2 s with the cup under the outlet, and closed at the end. The dispenser must move less than 2 mm. Arm-arm, arm-dispenser, arm-table, cup-dispenser and arm-stack contacts fail the run. Levels 2 and 3 also require the remaining stack to be standing and untouched.

**Why do the randomized runs fail?** The latest 24-trial table is 79 / 54 / 54 / 8 / 17 %. The common misses are the cup more than 15 mm from the coaster, the Level 4 dispenser creeping when the handle is pushed round, the Level 5 search pecking into the open mouth, and a grasp that closes on nothing.

**Has this run on the robot?** No. `get_robot(..., hardware=True)` raises unless `SIM_ALLOW_HARDWARE=1`, and this project does not set that variable.

**What would a later hardware test need?** A measured rim, a measured dish, a measured lever breakaway, and a run that is allowed to move the arms. Those measurements are not in this tree, and this tree does not invent them.
