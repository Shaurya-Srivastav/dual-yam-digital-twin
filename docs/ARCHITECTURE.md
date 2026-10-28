# Architecture

**Control loop.** `Plant` steps MuJoCo at 1 kHz. Each arm is a position actuator emulating the real MIT-mode law: `τ = kp(q*-q) + kd(q̇*-q̇) + gravity·factor`, with the feed-forward folded into the position target. Gains, gravity factors, friction, mass, base pose, sensor noise and command delay are randomized per trial (`Randomization.draw`). At power-up the runner holds two loaded poses and fits per-joint gravity feed-forward corrections from the position sag (`Session.calibrate_gravity`).

**Sensing.** Skills see encoders (q, q̇), motor torque, gripper width and effort, all with noise. `ContactMonitor` subtracts the controller's own gravity model and Coulomb friction from the motor torque and maps the residual to a Cartesian force with `-pinv(Jᵀ)`. Guards use force steps (`delta_force`) or a baseline-relative threshold, never a true contact flag.

**Skills and levels.** A skill is a generator: each `yield` is one 2 ms control tick. `Executor` runs a dependency graph over two arm lanes, fails loudly (`SkillFault`/`PlanError`), and records a per-step message that the UI shows as the check in words.

**Grasping.** The side grasp slides in with open jaws, complying laterally to contact force (admittance) so the funnel pads centre the hand; the jaws then close, and the stall width gives the chord `2·sqrt(r² - s²)`, i.e. how far off the cup's axis the hand is, so the depth is servoed until the close stalls near the cup's diameter.

**Referee.** `runner.check_pass` is the only code that reads true poses: cup within 10° of upright and 15 mm of the coaster centre, pour 7.0 ± 0.2 s with the cup under the outlet, handle closed, dispenser moved < 2 mm, no arm/dispenser/table/arm-arm contact, no arm link touching a cup or stack, stack still standing, arms at rest. An arm-arm contact aborts the run.

**Live twin.** One simulation thread (real time × speed), one render thread, one JPEG encoder thread. Frames go out over a WebSocket at up to 50 fps, capped at 1440×900 px and dropping to 8 fps when nothing moves; the browser decodes only the newest frame. Scrubbing replays the same seeded run to the chosen step.

**Dance.** `audio.py` decodes with ffmpeg, computes a log-mel onset envelope, estimates tempo, tracks beats with Ellis' dynamic programme and picks downbeats from bass energy. `dance.py` evaluates phase-locked moves at `beat_pos(t + lead)` and follows them with the same joint velocity/acceleration limits as the skills, with a 50 Hz collision query.
