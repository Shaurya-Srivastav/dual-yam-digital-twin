.PHONY: test-fast test-contracts test-nominal

export PYTHONPATH := .
export PYTHONUNBUFFERED := 1

test-fast:
	python3 -W ignore -u tests/test_offline.py

test-contracts:
	python3 -W ignore -u tests/test_contracts.py
	python3 -W ignore -u tests/test_numerics.py
	python3 -W ignore -u tests/test_planner_scene.py

test-nominal:
	python3 -W ignore -u -m sim_station.selftest
