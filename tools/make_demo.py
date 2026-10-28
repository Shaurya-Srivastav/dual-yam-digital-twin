"""Render a level headlessly into an animated GIF for the project page:  python tools/make_demo.py 1 site/demo_level1.gif"""
import sys

import mujoco
import numpy as np
from PIL import Image

import sim_station
from sim_station.runner import Session
from sim_station.runtime import CTRL_STEPS


def main(level: int, out: str, every_s: float = 0.6, w: int = 560, h: int = 350) -> None:
    S = Session()
    S.reset(level, "paper", seed=0)
    P = S.plant
    r = mujoco.Renderer(P.model, h, w)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = -28.0, -34.0, 1.45
    cam.lookat[:] = [0.42, 0.0, 0.08]
    r.update_scene(P.data, camera=cam); r.render()                     # first render on a fresh context is blank
    frames, nxt = [], 0.0
    while not S.ex.finished and not S.ex.fault and P.time - S.t_zero < 400:
        S.tick()
        t = P.time - S.t_zero
        if t >= nxt:
            nxt += every_s
            r.update_scene(P.data, camera=cam)
            frames.append(Image.fromarray(r.render()).convert("P", palette=Image.ADAPTIVE, colors=96))
    print(f"level {level}: {len(frames)} frames, finished={S.ex.finished}, pass={S.check_pass()['ok']}")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
