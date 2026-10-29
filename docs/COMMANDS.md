# Commands

Run these from the repository root with `PYTHONPATH` set to that root and `I2RT_PATH` pointing at a checkout of the arm meshes. `SIM_ASSETS_READY=1` skips rewriting meshes when they are already on disk.

| Command | What it checks | Last exit |
|---|---|---|
| `python3 tests/test_offline.py` | Geometry, path spec, seed partitions, click track. No MuJoCo. | See `docs/progress.md` for the run that produced the commit. |
| `python3 tests/test_numerics.py` | Wilson interval, belief walk, rim annulus, slerp, beat tracker. | Same. |
| `python3 tests/test_planner_scene.py` | Planner uses the belief, not the live cup or lever. | Same. |
| `python3 tests/test_contracts.py` | Graphs, JSON, API bounds, hardware lock, all five level graphs. | Same. |
| `python3 -m sim_station.selftest` | Nominal levels 1–5, pose-aligned replay, FK/IK self-check. Scene-as-reset is printed and does not affect the exit code. | Same. |
| `python3 -m sim_station.evaluate --levels 1 2 3 4 5 --trials 24 --jobs 8` | Randomized rates. All five levels in `docs/RESULTS.md` are the 2026-10-29 run, 24 trials, seeds 0–23. | |

`make test-fast` is the first row. `make test-nominal` is the selftest. None of these pass `--hardware`.

The GitHub workflow `.github/workflows/sim.yml` installs numpy and scipy and runs `tests/test_offline.py`. It has not been observed on GitHub from this machine. A green local run of that file is not a green GitHub run.
