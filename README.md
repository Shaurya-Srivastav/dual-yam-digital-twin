# Dual-YAM digital twin

A MuJoCo digital twin of a two-arm (I2RT YAM) water-cup station: pick a paper or plastic cup, hold it under a dry dispenser spout, let the second arm open the lever for a 7 s pour, and serve the cup into a coaster. Five levels of increasing uncertainty, all driven by one set of skills that only read what real motors would report (joint position, velocity, torque, gripper width and effort).

[![Dual-YAM digital twin: Level 1](https://img.shields.io/badge/project%20page-live-7c8cff)](https://dual-yam-digital-twin.netlify.app) ![license](https://img.shields.io/badge/license-MIT-4cd08a) ![sim](https://img.shields.io/badge/simulation%20only-no%20hardware%20run-f2c14e)

Videos of all five levels and of the dance mode are on the [project page](https://dual-yam-digital-twin.netlify.app); `tools/make_videos.py` regenerates them from the physics.

Project page: https://dual-yam-digital-twin.netlify.app

**Built on:** [Vezzani et al. 2017, tactile localisation](https://doi.org/10.1109/tro.2017.2707092) · [Murali, Dahiya, Kaboli 2022, information gain for tactile pose estimation](https://arxiv.org/abs/2205.04697) · [Brudermüller et al. ICRA 2026, touch-based localisation](https://doi.org/10.1109/icra57385.2026.11697596) · [Lee, Park, Cho 2025, contact-pause grasping](https://arxiv.org/abs/2509.12969) · [Lips and wyffels 2023, proprioceptive articulated manipulation](https://tlips.be/revisiting-proprioception-for-articulated-manipulation/) · [Jamali, Kormushev, Caldwell 2013, T-bar contact states](https://kormushev.com/papers/Jamali_OCEANS-2013.pdf) · [Toskov et al. CoRL 2022, gravitational pivoting](https://arxiv.org/abs/2210.05068). What each one contributed, and what this twin does not do from it, is in [docs/RESEARCH.md](docs/RESEARCH.md).

## What is in here

| Part | Where | What it does |
|---|---|---|
| Calibration | `sim_station/layout.py` | Scene numbers in one place. Provenance is in `docs/ASSUMPTIONS.md`. Includes the recorded Level 1 and Level 2 waypoints |
| World | `scene.py`, `plant.py` | MuJoCo scene (hollow stave cups, nested stack, passive 360° lever with a lift hinge, funnel coaster), MIT-style PD control with gravity feed-forward, noisy sensors, contact referee |
| Sensing | `sensing.py` | Contact force from torque residuals: `F = -pinv(Jᵀ)(τ - τ_gravity - τ_friction)` |
| Motion | `kin.py`, `motion.py` | Strict mink IK, min-jerk synchronized trajectories, Cartesian lines |
| Skills | `skills*.py` | Side grasp with lateral admittance and chord-width depth servo, spout insertion, lever routine, coaster place, pecks, handle search, one-arm in-air cup flip |
| Levels | `levels.py` | Dependency graphs of skills on two arm lanes |
| Proofs | `proofs.py`, `evaluate.py`, `selftest.py` | Real-data replay, IK vs real poses, Monte Carlo with a failure histogram |
| Live twin | `station.py`, `server.py`, `static/index.html` | 40+ fps JPEG stream, view cube, timeline with scrub, telemetry, Monte Carlo, Dance Mode |
| Dance | `audio.py`, `dance.py` | Beat tracking of your own audio files and beat-locked choreography |
| Simulator driver | `sim_robot.py` | `SimYam` implements the same method names as i2rt's robot object. `hardware=True` raises unless `SIM_ALLOW_HARDWARE=1`. This repo does not start the arms |

## Run it

```bash
git clone https://github.com/Shaurya-Srivastav/dual-yam-digital-twin && cd dual-yam-digital-twin
git clone https://github.com/i2rt-robotics/i2rt third_party/i2rt     # arm and gripper models (or set I2RT_PATH)
pip install -r requirements.txt            # Python 3.11, macOS or Linux; ffmpeg needed for Dance Mode
./run_sim.sh                               # http://127.0.0.1:8090
python -m sim_station.selftest             # nominal levels + pose-aligned replay. Not the randomized rates. See docs/STATUS.md
python -m sim_station.evaluate --levels 1 2 3 4 5 --trials 50 --jobs 8
python -m sim_station.proofs               # replay of the real Level 1 waypoints + IK equivalence
```

Dance Mode plays three songs from `sim_station/assets/*.mp3`. Owning a copy of a song does not give the right to republish it, so the recordings are not in this public repo, its videos or the project page: the owner keeps them in a private repo and fetches them with `tools/get_audio.sh`; anyone else adds their own mp3s under the file names in `sim_station/audio.py` (`SONGS`). On the project page, "Play my own copy" plays a song file from the visitor's computer in sync with the dance clip without uploading it. Beats are tracked from the audio itself and cached as numbers only.

## The five levels

1. Upright cup in the coaster, position known.
2. Inverted nested stack on the coaster spot, height known: pin the lower stack with the tap arm, turn the cup hand into the flip attitude, grasp the top cup, wiggle, pull, **roll it upright with one arm** (see below), set it down, side-grasp, serve.
3. Same, height unknown (3 to 11 cups): peck down on the stack to count the cups.
4. Lever angle unknown (it turns through a full 360°): sweep a fingertip around the valve, stop at the first touch (below the handle's breakaway force), compute the handle yaw from the known hand and handle shapes, park it, touch again to confirm.
5. Cup anywhere on the mat: coarse peck grid, then a set-membership fit of where the cup must stand from every peck outcome. The next peck is the one that splits the remaining candidate centres about in half (not an expected-entropy computation).

### One-arm cup flip

The wrist is turned to the flip attitude, about −65° off the arm's radial line and 8° down, **before** the fingers close on the top cup's base band. The lift off the stack, a 9 cm slide clear of it and one Cartesian roll of about 166° turn the cup upright. The wrist housing cannot follow the cup base the last few centimetres, so the cup is let go a short height above the mat and lands on its base. The attitude was found by an IK search over hand attitudes (`skills_l2.flip_motion`); it is not the gravitational pivoting of Toskov et al.

## Honest status

See [docs/RESULTS.md](docs/RESULTS.md). Short version: the nominal (no randomization) runs of all five levels pass the full referee, with the remaining stack never touched in Levels 2 and 3. The real-data replay passes (cup 0.9 mm from the outlet axis). The randomized success rates (24 trials per level) are 79 / 42 / 42 / 21 / 12 %, well below the 98 / 98 / 95 / 95 / 90 % targets. Nothing here has run on the real robots.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): control loop, sensing, skills, referee
- [docs/RESULTS.md](docs/RESULTS.md): measured numbers and open problems
- [docs/STATUS.md](docs/STATUS.md): which claims are measured, simulated, or not run
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): which numbers were recorded, derived, assumed, or fitted
- [docs/progress.md](docs/progress.md): which offline-brief items have a command and an exit code
- [docs/FRONTIER_ROADMAP.md](docs/FRONTIER_ROADMAP.md): what is still open, in order
- [docs/RESEARCH.md](docs/RESEARCH.md): the literature behind the search and flip methods
- [docs/ABSTRACT.md](docs/ABSTRACT.md), [docs/QA.md](docs/QA.md), [docs/COMMANDS.md](docs/COMMANDS.md), [docs/LICENSES.md](docs/LICENSES.md)

The twin does not use cameras anywhere. Skills never read true object poses; only the referee (`runner.check_pass`) and the debug overlay do.

## Repository layout

```
sim_station/      the twin (see the table above)
tools/            make_videos.py (project-page videos), get_audio.sh
site/             static project page deployed to Netlify (index.html + videos/)
docs/             ARCHITECTURE, RESULTS, RESEARCH, ROADMAP
```

## Contributing

Run `python -m sim_station.selftest` before opening a pull request; it must keep the real-data replay and the IK equivalence passing. Anything that changes behaviour should come with a before/after Monte Carlo (`python -m sim_station.evaluate`) in the description.
