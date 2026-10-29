# Scope

This repository is a simulation-only benchmark. The work in `docs/progress.md` does not use a robot, a CAN bus, a Raspberry Pi, or any new physical measurement.

Dimensions in `sim_station/layout.py` stay as they are. The 76 mm rim is the nominal model. An earlier note of about 90 mm is an unmeasured alternative, not a number to silently substitute. The tap-base offset, dish radii and valve shape are assumptions carried by the existing scene. Where a real value is missing, the result is marked not verified on hardware.

The pour is a dry timer. Contact force is a torque residual, not a calibrated load cell. A nominal pass is not a claim that the same motion would succeed on the arms.

Publication of station-derived waypoint arrays is unchanged from the tree the owner already published. This work does not add logs, photos, credentials, or song files.
