"""Solver settings written into the MJCF. These are the simulator's settings, not a calibration."""

OPTION = dict(
    timestep=0.001,
    gravity="0 0 -9.81",
    integrator="implicitfast",
    cone="elliptic",
    impratio=8,
    solver="Newton",
    iterations=40,
    tolerance=1e-9,
    noslip_iterations=3,
    noslip_tolerance=1e-7,
)

NOTE = "Simulation settings. The pour is a dry timer. Stack upright torque and the nested-cup tether are synthetic."
