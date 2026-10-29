# Licenses and what is not in this tree

This repository is MIT, copyright 2026 Shaurya Srivastav. See `LICENSE`.

The arm and gripper meshes come from the I2RT YAM description, loaded through `I2RT_PATH` or `third_party/i2rt`. That checkout is not vendored here. Its license is the upstream project's, not this MIT grant.

MuJoCo, NumPy, SciPy, FastAPI, mink and the other packages in `requirements.txt` stay under their own licenses. `pyproject.toml` names the same direct dependencies. Versions are whatever the local environment has; this tree does not claim a lockfile was solved on a clean machine.

The three named songs are not in the public tree, the public videos, or the project page. A generated click track is available in the local dashboard as song id `click`. It is synthesized in `audio.click_track`. It is not a recording.

Waypoint arrays already published in `layout.py` stay as they were. This work does not add station photos, credentials, or logs. `tools/scan_tracked.py` looks for private-key blocks, cloud key ids, and password-shaped assignments in tracked text. A clean run means none of those patterns matched. It is not a legal review.
