#!/bin/bash
# Start the dashboard on http://127.0.0.1:8090 (set PORT to change it, I2RT_PATH to point at your i2rt clone)
cd "$(dirname "$0")"
export PORT="${PORT:-8090}"
exec python3 -m sim_station
