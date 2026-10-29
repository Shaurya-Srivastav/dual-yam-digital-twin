# Progress against the offline completion brief

HEAD this branch started from `b300789` (parent of the audit snapshot `32b5d08`). The 408-item brief is the contract. It is not finished. Items below are the ones with a command and an exit code. Everything else is not done.

The brief's line that `collisions_for` ignores cups was already fixed on `b300789`. The planner places the cup, stack, coaster and lever from the belief. Re-checked in this branch; that code is still there.

## Done on this branch, with a test

| Item | What changed | Evidence |
|---|---|---|
| P06 contact audit | The referee records contacts on every physics step, not every third. | `python -m sim_station.selftest` exit 0 after the change. Nominal cups 4, 12, 12, 3, 11 mm. Thresholds unchanged. |
| P07 coaster | The place step does not walk back to the believed coaster when the dish pushed the other way. | `tests/test_numerics.py` exit 0. Nominal Level 1 still 4 mm. Randomized rate not rerun. |
| P11 mouth | A peck that misses at 8 cm does not delete cup centres inside the mouth. A coarse miss is followed by two pecks 32 mm away. Level 5 copy says "search region", not the whole mat. | `test_mouth_miss_is_not_a_rim_miss`. Nominal Level 5 passes, cup 11 mm. |
| P05 slerp | A 180 degree orientation no longer divides by a zero sine. | `test_slerp_near_pi_is_a_rotation`. |
| P13 stats | Wilson 95% interval, pass-only mean time labeled, all-trial mean time, every failure reason as well as the first. | `test_wilson_contains_rate`. No new Monte Carlo in this commit. |
| P17 beat gate | Removed the dead `or True`. The tracker takes the best finite predecessor. | `test_click_track_bpm`, `test_silence_does_not_crash`. |
| P08 dead branch | Removed a distance check that could never be true (`norm < 0`). | The pre-rotated flip path is unchanged. |
| P15 HTML | Event text, step titles, Monte Carlo reasons and proof errors are escaped. Unknown lever yaw renders as "unknown". | Source inspection. No browser run in this commit. |
| Task graph | `validate_steps` rejects a cycle, a duplicate id, and a missing dependency. `Executor` calls it. | `tests/test_contracts.py` exit 0, including `Session.reset` for levels 1–5. |
| API bounds | Reset, speed, trial count and song id are checked before the station runs. Bad input returns HTTP 422. | `test_api_bounds`. The server module was compiled, not started in a browser. |
| JSON | WebSocket snapshots and `evaluate --json` go through `json_ready`. NaN becomes null. A sibling `.jsonl` has one row per trial. | `test_json_ready_drops_nan`, `test_report_writer`. |
| Hardware gate | `get_robot(..., hardware=True)` raises unless `SIM_ALLOW_HARDWARE=1`. The flag was not set. | `test_hardware_stays_off`. |
| Skill boundary | The three skill modules do not call `cup_state` or `outlet_xy_true`. The flip slip note uses grip force and width only. | `test_skills_do_not_read_cup_pose`. Nominal Level 2 still passes. |
| Short path | `cart_path` raises if given fewer than two poses. | `test_short_cartesian_path`. |
| Stack assist | `Plant.stack_assist` defaults on. Off skips the synthetic upright torque. | `test_stack_assist_can_be_turned_off`. No physics ablation was run. |
| Assumptions | `sim_station/assumptions.py` and `docs/ASSUMPTIONS.md` list recorded, derived, assumed and fitted numbers. | `test_assumptions_name_real_attributes`. |
| Search geometry | A 6 cm grid can miss a 38 mm rim radius. The fit stores the radius of the centres still consistent with the pecks. | `tests/test_offline.py` `test_grid_does_not_cover_the_rim`, `test_feasible_region_radius`. Nominal Level 5 still passes. |
| Path spec | Consecutive duplicate poses are dropped. A path with no motion raises. A pure rotation is kept. | `test_repeated_pose_is_rejected_and_a_turn_is_kept`. `cart_path` turns that into `PlanError`. |
| Leaving pairs | A contact that is present at the start of a planned path stops counting once the pair separates. | `test_leaving_expires_after_separation`. Nominal selftest still exit 0. |
| Frames and actions | Pose inverse round-trips. A replay packet past the end is marked stale. A non-finite joint command is rejected. | `test_pose_round_trip`, `test_replay_goes_stale_and_limits_reject_nan`. |
| Rim scenario | The 76 mm nominal rim fits the 77.6 mm jaw. A hypothetical 90 mm rim does not, and is not substituted. | `test_nominal_rim_fits_and_90mm_does_not`. |
| Seed partitions | Dev, test and stress seeds do not overlap seeds 0–23. No rate is quoted for the held-out seeds. | `test_seed_partitions_do_not_overlap_the_published_window`. |
| CI | `.github/workflows/sim.yml` installs numpy and scipy and runs `tests/test_offline.py`. It does not pass `--hardware` or `SIM_ALLOW_HARDWARE`. | `test_workflow_cannot_start_the_arms`. The workflow has not been run on GitHub from this branch. |
| Scheduler | A global step waits until both arm lanes are free. An arm step does not start during a global step. | `test_global_step_does_not_share_an_arm`. Nominal selftest still passes. |
| Audio edges | Audio shorter than one window, and non-finite samples, return no tempo instead of raising. Song id `click` is a generated click track with a WAV body and a dance plan. | `test_short_and_nonfinite_audio_do_not_crash`, `test_click_track_is_a_wav_and_has_a_dance`. |
| Camera and pick | Non-finite camera and pick values are rejected before the station runs. | `test_camera_and_pick_reject_nonfinite`. |
| Force sign | A positive joint-torque residual maps to a force into the hand. A zero Jacobian is not marked reliable. The monitor uses that function. Skills do not yet refuse a contact because of it. | `test_torque_residual_sign`. |
| Solver record | The MJCF option block is `physics_manifest.OPTION`. | `test_physics_option_is_the_documented_solver`. Selftest loaded the scene after the change. |

## Already done before this branch (`b300789`)

Nominal Levels 1–5 pass the strict referee. Pose-aligned replay passes (0.9 mm / 8.9 mm). Scene-as-reset replay fails (105.7 mm / 19.7 mm) and is labeled as such. Level 4 belief yaw serializes as null. Planner tests in `tests/test_planner_scene.py`.

## Re-checked after the path, scheduler, and sensing edits

These commands were run on this working tree. Exit code 0 unless noted.

- `python3 tests/test_offline.py` — geometry, schedules, click-track WAV, short audio, camera bounds, seed partitions.
- `python3 tests/test_numerics.py`
- `python3 tests/test_contracts.py` — includes the torque-residual sign.
- `python3 tests/test_planner_scene.py`
- `python3 tools/scan_tracked.py` — no secret-shaped lines in tracked text.
- `python3 -m sim_station.selftest` — levels 1–5 pass, cups 4 / 12 / 12 / 3 / 11 mm, flows 6.96 / 6.96 / 6.95 / 6.95 / 6.95 s. Pose-aligned replay 0.9 mm and 8.9 mm. Scene-as-reset replay fails at 105.7 mm and 19.7 mm and does not affect the exit code. FK/IK self-consistency passes. Wall time about 119 s.

A global step now waits until both arm lanes are free, and an arm step does not start while a global step is running. The nominal pour step already waited on the previous arm steps, and the selftest still passes. The solver line in the MJCF is `sim_station/physics_manifest.py`. The force sign is the same formula as before, called from `world_force_from_tau`.

## Motion retest after the one-arm roll back

Nominal seed 0, paper, `Session.run` then `check_pass`. Not a Monte Carlo.

| Level | Result | Cup | Flow | Sim time |
|---|---|---|---|---|
| 1 | pass | 3.4 mm | 6.95 s | 55 s |
| 2 | pass | 4.9 mm | 6.95 s | 102 s |
| 3 | pass | 4.5 mm | 6.95 s | 109 s |
| 4 | pass | 4.1 mm | 6.95 s | 71 s |
| 5 | pass | 13.8 mm | 6.94 s | 76 s |

In Levels 2 and 3 the cup hand comes in with the wrist rolled half a turn the other way and closes on the middle of the inverted top cup. After the pull it rolls the wrist back to its normal position, so the cup comes out upright in the ordinary pour grip and goes straight to the spout. One arm holds the cup from the stack to the coaster; there is no hand-off and no set-down. It is an in-hand roll, not a free tumble. The grasp dips 8° (a level hand-up grasp puts J5 on its stop), so the cup leans 16° after the roll until the carry pitches the hand. In the sim the nested cups are tethered and do not collide, so squeezing the top cup at its middle does not also squeeze the cup inside it; on real cups that coupling is untested. Level 4 closes on the centre of the knob, then closes around a circle until the handle stalls in the jaws. On this seed the handle is already at 45°. Level 5 pecks with the jaws at 62 mm. The wrist stays steep. The slide under the spout still dips 4 mm when the wrist can; the corridor is chosen so that dip is reachable.

## Not done

Randomized rates were not rerun, so 79 / 42 / 42 / 21 / 12 % is still the last measured table, on `32b5d08`, not a claim about the one-arm roll back or the circle search. Physics ablations of the stack torque and the cup tether were not run (`stack_assist` can be turned off; that run was not done). No 100-trial benchmark. No behavior cloning and no reinforcement learning. No hardware. No new physical measurements. `.github/workflows/sim.yml` and `pyproject.toml` are in the tree; the dependencies are not pinned to a lock solved on a clean machine. Scene-as-reset replay still fails. The flip is still an in-hand roll. The songs stay off the public site.
