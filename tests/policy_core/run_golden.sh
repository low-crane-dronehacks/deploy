#!/usr/bin/env bash
# Golden test of the C++ policy core against the Python implementation (offline; host side, Docker, no network).
#   tests/policy_core/run_golden.sh <team_ws> <scratch with depth_vpi.npz + depth_sgbm.npz> <out dir> [policy ...]
# Needs a built workspace (team_build.sh). Generates golden_<policy>.npz with Python, then replays it in C++.
set -euo pipefail
WS=${1:?team_ws (the directory with src/deploy and install/)}
DATA=${2:?directory with depth_vpi.npz and depth_sgbm.npz}
OUT=${3:?output directory for the golden files}
shift 3
POLICIES=("${@:-cur4_clean50_800 comp_s2}")
read -ra POLICIES <<< "${POLICIES[*]}"
IMAGE=${IMAGE:-skybeta:0.3.2}
mkdir -p "$OUT"
docker run --rm --network none -e OPENBLAS_NUM_THREADS=1 -v "$WS":/root/team_ws:ro -v "$DATA":/data:ro \
  -v "$OUT":/out "$IMAGE" bash -lc '
  set -e
  for p in '"${POLICIES[*]}"'; do
    python3 -I /root/team_ws/src/deploy/tests/policy_core/make_golden.py /root/team_ws/src/deploy/policies/$p.npz \
      /out/golden_$p.npz /data/depth_vpi.npz /data/depth_sgbm.npz
    /root/team_ws/install/policynav/lib/policynav/policy_core_golden_test /out/golden_$p.npz \
      /root/team_ws/src/deploy/policies/$p.npz
  done
  /root/team_ws/install/policynav/lib/policynav/policy_core_decision_gate_test'
