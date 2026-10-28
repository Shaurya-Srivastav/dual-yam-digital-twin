"""Honest sensing: the runner sees only encoder position/velocity, motor torque feedback and gripper width/effort.

`ContactMonitor` turns torque feedback into an end-effector force estimate (measured motor torque minus the
nominal gravity model, mapped through the Jacobian). Guarded moves trigger on that estimate only.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from sim_station import layout as L
from sim_station.kin import ArmKin
from sim_station.plant import ArmObs, Plant


class ContactMonitor:
    def __init__(self, arm: str, plant: Plant, kin: ArmKin, cutoff_hz: float = 22.0, dt: float = L.CONTROL_DT) -> None:
        self.arm, self.plant, self.kin = arm, plant, kin
        self.alpha = dt / (1.0 / (2.0 * np.pi * cutoff_hz) + dt)
        self.tau_ext = np.zeros(6)
        self.force = np.zeros(3)           # world-frame force estimate at the TCP (N)
        self.moment = np.zeros(3)
        self.q = np.zeros(6)
        self.qd = np.zeros(6)
        self.obs: Optional[ArmObs] = None
        self.tcp = np.eye(4)
        self.grip_width = L.GRIPPER_MAX_OPEN_M
        self.grip_force = 0.0
        self.grip_vel = 0.0
        self._n = 0
        self.fbuf: deque = deque(maxlen=96)

    def reset(self, obs: ArmObs) -> None:
        self.update(obs, settle=True)

    def update(self, obs: ArmObs, settle: bool = False) -> None:
        self.obs = obs
        self.q = obs.q
        self.qd = obs.qd
        self.grip_width, self.grip_vel, self.grip_force = obs.grip_width, obs.grip_vel, obs.grip_force
        g = self.plant.gravity_model(self.arm, obs.q)
        # Coulomb friction is part of the model (the real controller exposes the same constants); remove it so a
        # joint changing direction does not look like a collision
        fric = L.JOINT_FRICTIONLOSS * np.tanh(obs.qd / 0.004)
        raw = obs.tau - g - fric
        a = 1.0 if settle else self.alpha
        self.tau_ext += a * (raw - self.tau_ext)
        self._n += 1
        if self._n % 2 == 0 or settle:
            self.tcp = self.kin.fk(obs.q)
            J = self.kin.jacobian(obs.q)
            Jp = J[:3]
            A = Jp @ Jp.T + 1e-3 * np.eye(3)
            # tau_ext = motor torque - gravity torque = -J^T F_env at equilibrium, so the force the world puts on the hand
            # is the negative of the least-squares solution. The Jacobian lives in the arm base frame.
            self.force = -(self.kin.R_wb @ np.linalg.solve(A, Jp @ self.tau_ext))
            self.fbuf.append(self.force.copy())

    def delta_force(self, ticks: int = 25) -> np.ndarray:
        """Change of the contact-force estimate over the last `ticks` force samples. Friction and gravity-model offsets are
        constant over a short window and cancel; a collision shows up as a step."""
        if len(self.fbuf) <= ticks:
            return np.zeros(3)
        return self.fbuf[-1] - self.fbuf[-1 - ticks]

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.tcp[:3, 3]


class Baseline:
    """Free-space force statistics collected while moving in the same way the guarded move will."""

    def __init__(self) -> None:
        self.samples = []

    def add(self, f: np.ndarray) -> None:
        self.samples.append(np.array(f, dtype=float))

    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.samples) < 5:
            return np.zeros(3), np.full(3, 0.3)
        a = np.array(self.samples)
        return a.mean(axis=0), a.std(axis=0) + 1e-3

    def reset(self) -> None:
        self.samples.clear()
