import os
import sys

# MuJoCo's offscreen renderer on macOS needs this before mujoco is imported.


from sim_station.server import main

if __name__ == "__main__":
    main()
