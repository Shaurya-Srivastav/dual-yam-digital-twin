#!/bin/bash
# Pull the three Dance Mode songs from the owner's PRIVATE audio repo into sim_station/assets/ (needs gh/git access to it).
# Anyone else: put your own mp3s there with the file names listed in sim_station/audio.py (SONGS).
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/Shaurya-Srivastav/everest-dual-yam-twin-audio "$tmp"
cp "$tmp"/*.mp3 sim_station/assets/
rm -rf "$tmp"
echo "songs copied to sim_station/assets/"
