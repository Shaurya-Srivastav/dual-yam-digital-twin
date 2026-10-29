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

## Already done before this branch (`b300789`)

Nominal Levels 1–5 pass the strict referee. Pose-aligned replay passes (0.9 mm / 8.9 mm). Scene-as-reset replay fails (105.7 mm / 19.7 mm) and is labeled as such. Level 4 belief yaw serializes as null. Planner tests in `tests/test_planner_scene.py`.

## Not done

Randomized rates were not rerun, so 79 / 42 / 42 / 21 / 12 % is still the last measured table, on `32b5d08`, not a claim about this branch. Physics ablations of the stack torque and the cup tether were not run. No 100-trial benchmark. No Playwright run. No behavior cloning and no reinforcement learning. No hardware. No new measurements. CI, packaging pins, and a clean-room install were not added. The public site is not redeployed from this branch.
