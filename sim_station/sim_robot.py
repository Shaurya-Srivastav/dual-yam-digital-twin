"""One driver for both worlds: `SimYam` speaks i2rt's `Robot` protocol on top of the MuJoCo plant.

    robot = get_robot("cup", hardware=False)            # the twin
    robot = get_robot("cup", hardware=True, channel="can0")   # the real arm, at half speed on first runs

Both return an object with the same methods (`get_joint_pos`, `command_joint_pos`, `get_observations`, ...), so a skill or a test
trajectory written against the protocol runs unchanged on either. `python -m sim_station.sim_robot --hardware` plays three
standard test trajectories on the twin and, when the arms are connected, on the hardware, and prints how far apart the
tracking errors are: the numbers to fit friction and gravity factors against once the Pi is back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from sim_station import layout as L
from sim_station.plant import ARMS, Plant
from sim_station.runtime import CTRL_STEPS

GRIP_SPAN = 0.0776                      # jaw gap at gripper command 1.0 (calibrated spans 5.31 / 5.26 rad on a 6.57 rad = 96 mm stroke)


class SimYam:
    """i2rt `Robot` protocol on one arm of the plant. 7 DoF: six joints plus the gripper as a 0..1 command (1.0 = fully open)."""

    def __init__(self, plant: Optional[Plant] = None, arm: str = "cup", control_period_s: float = 0.01, grip_cap_n: float = 20.0) -> None:
        self.plant = plant or Plant()
        if plant is None:
            self.plant.reset(1, "paper")
        self.arm = arm
        self.period = control_period_s
        self.cap = grip_cap_n
        self._q_cmd = self.plant.arm_q(arm).copy()
        self._g_cmd = 1.0

    # ------------------------------------------------------------- Robot protocol
    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        o = self.plant.read_arm(self.arm)
        return np.append(o.q, o.grip_width / GRIP_SPAN)

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        o = self.plant.read_arm(self.arm)
        return {"pos": np.append(o.q, o.grip_width / GRIP_SPAN), "vel": np.append(o.qd, o.grip_vel / GRIP_SPAN)}

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        jp = np.asarray(joint_pos, dtype=float)
        if jp.shape[0] not in (6, 7):
            raise ValueError("expected 6 joints (+ gripper)")
        self._q_cmd = jp[:6].copy()
        if jp.shape[0] == 7:
            self._g_cmd = float(np.clip(jp[6], 0.0, 1.0))
        self.plant.command_arm(self.arm, self._q_cmd, np.zeros(6))
        self.plant.command_gripper(self.arm, self._g_cmd * GRIP_SPAN, cap_n=self.cap)
        self._advance(self.period)

    def command_target_vel(self, joint_vel: np.ndarray) -> None:
        v = np.asarray(joint_vel, dtype=float)[:6]
        self._q_cmd = self._q_cmd + v * self.period
        self.plant.command_arm(self.arm, self._q_cmd, v)
        self._advance(self.period)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        pos = np.asarray(joint_state["pos"], dtype=float)
        self._q_cmd = pos[:6].copy()
        if pos.shape[0] == 7:
            self._g_cmd = float(np.clip(pos[6], 0.0, 1.0))
        self.plant.command_arm(self.arm, self._q_cmd, np.asarray(joint_state.get("vel", np.zeros(6)), dtype=float)[:6])
        self.plant.command_gripper(self.arm, self._g_cmd * GRIP_SPAN, cap_n=self.cap)
        self._advance(self.period)

    def get_observations(self) -> Dict[str, np.ndarray]:
        o = self.plant.read_arm(self.arm)
        return {
            "joint_pos": o.q.copy(), "joint_vel": o.qd.copy(), "joint_eff": o.tau.copy(),
            "gripper_pos": np.array([o.grip_width / GRIP_SPAN]), "gripper_vel": np.array([o.grip_vel / GRIP_SPAN]),
            "gripper_eff": np.array([o.grip_force]),
        }

    def joint_pos_spec(self):
        from dm_env.specs import Array
        return Array(shape=(7,), dtype=np.float32)

    def joint_state_spec(self):
        from dm_env.specs import Array
        return {"pos": Array(shape=(7,), dtype=np.float32), "vel": Array(shape=(7,), dtype=np.float32)}

    def get_robot_type(self):
        from i2rt.robots.robot import RobotType
        return RobotType.ARM

    def get_robot_info(self) -> Dict[str, object]:
        return {"kp": L.KP_ARM.copy(), "kd": L.KD_ARM.copy(), "gravity_comp_factor": L.GRAV_COMP_FACTOR_REAL.copy(),
                "joint_limits": np.stack([L.JOINT_LOW, L.JOINT_HIGH]), "gripper_span_m": GRIP_SPAN, "sim": True}

    def reinit(self) -> None:
        self._q_cmd = self.plant.arm_q(self.arm).copy()

    def close(self) -> None:
        pass

    # ---------------------------------------------------------------- helpers
    def _advance(self, seconds: float) -> None:
        n = max(int(round(seconds / (CTRL_STEPS * self.plant.dt))), 1)
        for _ in range(n):
            self.plant.step(CTRL_STEPS)

    def advance(self, seconds: float) -> None:
        self._advance(seconds)


class SpeedLimited:
    """Wrap any Robot so no commanded joint moves faster than `scale` of the vendor limits (50% on first hardware runs)."""

    def __init__(self, robot, scale: float = 0.5, period_s: float = 0.01) -> None:
        self.r, self.scale, self.dt = robot, scale, period_s
        self._last: Optional[np.ndarray] = None

    def __getattr__(self, name):
        return getattr(self.r, name)

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        jp = np.asarray(joint_pos, dtype=float).copy()
        cur = self._last if self._last is not None else np.asarray(self.r.get_joint_pos(), dtype=float)
        lim = np.append(L.MAX_JOINT_VEL, 1.0) * self.scale * self.dt
        jp = cur + np.clip(jp - cur, -lim[: jp.shape[0]], lim[: jp.shape[0]])
        self._last = jp
        self.r.command_joint_pos(jp)


def get_robot(arm: str = "cup", hardware: bool = False, channel: Optional[str] = None, plant: Optional[Plant] = None, speed_scale: float = 0.5):
    """The same call for both worlds. Hardware needs the i2rt package and the CAN interface."""
    if not hardware:
        return SimYam(plant, arm)
    sys.path.insert(0, os.environ.get("I2RT_PATH", "third_party/i2rt"))
    from i2rt.robots.get_robot import get_yam_robot            # noqa: E402

    ch = channel or ("can0" if arm == "cup" else "can1")
    return SpeedLimited(get_yam_robot(channel=ch, zero_gravity_mode=False), scale=speed_scale)


# ------------------------------------------------------------------ standard tracking tests
TEST_TRAJECTORIES = {
    "shoulder_swing": [np.array([0.0, 1.0, 0.6, 0.2, 0.0, 0.0, 1.0]), np.array([0.5, 1.4, 1.0, 0.6, 0.0, 0.0, 1.0]), np.array([-0.3, 1.2, 0.8, 0.4, 0.0, 0.0, 1.0])],
    "elbow_wrist": [np.array([0.0, 1.2, 1.1, 0.3, 0.0, 0.0, 1.0]), np.array([0.0, 1.2, 0.5, 0.9, 0.4, 0.8, 1.0]), np.array([0.0, 1.2, 1.1, 0.3, -0.4, -0.8, 1.0])],
    "gripper_cycle": [np.array([0.0, 1.2, 1.1, 0.3, 0.0, 0.0, 1.0]), np.array([0.0, 1.2, 1.1, 0.3, 0.0, 0.0, 0.3]), np.array([0.0, 1.2, 1.1, 0.3, 0.0, 0.0, 1.0])],
}


def run_tracking(robot, name: str, seconds_per_leg: float = 2.0, period: float = 0.01) -> Dict[str, List[float]]:
    """Min-jerk legs between the waypoints at the robot's own control period; returns commanded vs measured joint traces."""
    wps = TEST_TRAJECTORIES[name]
    start = np.asarray(robot.get_joint_pos(), dtype=float)
    legs = [start] + wps
    cmd, meas, t = [], [], []
    clock = 0.0
    for a, b in zip(legs[:-1], legs[1:]):
        n = int(seconds_per_leg / period)
        for k in range(1, n + 1):
            s = k / n
            s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            q = a + (b - a) * s
            robot.command_joint_pos(q)
            if not hasattr(robot, "advance"):
                time.sleep(period)
            cmd.append(q.tolist())
            meas.append(np.asarray(robot.get_joint_pos(), dtype=float).tolist())
            clock += period
            t.append(clock)
    return {"t": t, "cmd": cmd, "meas": meas}


def tracking_error(trace: Dict[str, List[float]]) -> np.ndarray:
    return np.sqrt(np.mean((np.array(trace["cmd"]) - np.array(trace["meas"])) ** 2, axis=0))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", action="store_true", help="also run the trajectories on the real arm (half speed)")
    ap.add_argument("--arm", default="cup", choices=list(ARMS))
    ap.add_argument("--out", default="tracking_report.json")
    a = ap.parse_args(argv)
    report: Dict[str, dict] = {}
    twin = get_robot(a.arm, hardware=False)
    for name in TEST_TRAJECTORIES:
        rms_sim = tracking_error(run_tracking(twin, name))
        entry = {"sim_rms_rad": rms_sim.round(5).tolist()}
        if a.hardware:
            hw = get_robot(a.arm, hardware=True)
            try:
                rms_hw = tracking_error(run_tracking(hw, name))
                entry["hardware_rms_rad"] = rms_hw.round(5).tolist()
                entry["ratio_hw_over_sim"] = (rms_hw / np.maximum(rms_sim, 1e-6)).round(2).tolist()
            finally:
                hw.close()
        report[name] = entry
        print(name, json.dumps(entry))
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    if a.hardware:
        print("Fit: raise gravity_comp_factor where hardware sags more than the twin; raise Coulomb friction where it lags more.")


if __name__ == "__main__":
    main()
