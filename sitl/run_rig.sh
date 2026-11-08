#!/bin/bash
# PX4 SITL rig for the skybeta drone: PX4 v1.16.2 (sitl/PX4-Autopilot) + Gazebo Harmonic + ROS 2 Humble, running
# skybeta's own flight_interface and px4_odom_bridge with our team launch. Same on the Jetson and on an x86 lab PC.
#
#   run_rig.sh build [image|px4|ws|all]   toolchain image, PX4 SITL, the ROS workspace (default: all, in that order)
#   run_rig.sh up                         start the rig (detached) and wait until PX4 + the sim feeds are READY
#   run_rig.sh team [launch args]         our launch, as the stack's competitor window starts it:
#                                         ros2 launch policynav skybeta.launch.py stereo_backend:=none ...
#                                         TEAM_LAUNCH=skybeta_climb_forward.launch.py run_rig.sh team fly:=true
#   run_rig.sh pilot status|fly [--arm]   scripted safety pilot (sitl/pilot.py); `fly` without --arm is a dry run
#   run_rig.sh sh | logs | down
#
# Arming in SITL follows the same rule as the drone (AGENTS.md): only with the team's OK for that run.
#
# Environment (up): SPAWN_POSE=x,y,z,roll,pitch,yaw_enu  BATTERY_SOC=0..1  CAMERA_X/Y/Z/ROLL/PITCH/YAW (drone.env keys)
#   PX4_PARAMS_DUMP=<file under sitl/params/>  VIO_LATENCY_MS VIO_SIGMA_P VIO_DRIFT_P VIO_DRIFT_YAW VIO_OUTAGE_AT_S
#   VIO_OUTAGE_S  DEPTH_NOISE_K DEPTH_DROPOUT  SITL_DOMAIN_ID (150)  SKYBETA_DIR (skybeta checkout)  JOBS (build)
#
# Isolation: the rig runs in its own network and IPC namespace (docker's default bridge, private /dev/shm), DDS stays on
# the container's loopback (skybeta's fastdds_localhost.xml / agent_refs_local.xml), and the domain (150) is outside the
# fleet's 1..99. So it is safe to run on a drone whose real stack is up: nothing here can reach the real agent or FC.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
PX4="$HERE/PX4-Autopilot"; CACHE="$HERE/.cache"
IMAGE=dronehacks-rig:humble-px4-v1.16.2; NAME=dronehacks-rig
DOMAIN="${SITL_DOMAIN_ID:-150}"; JOBS="${JOBS:-4}"
PX4_TAG=v1.16.2
die() { echo "[rig] ERROR: $*" >&2; exit 1; }

# skybeta checkout (read-only): ../skybeta next to deploy (dev layout), ~/skybeta on a drone
SKYBETA="${SKYBETA_DIR:-}"
if [[ -z "$SKYBETA" ]]; then
  for d in "$REPO/../skybeta" "$HOME/skybeta"; do [[ -f "$d/host/dds/agent_refs_local.xml" ]] && SKYBETA="$(cd "$d" && pwd)" && break; done
fi
[[ -n "$SKYBETA" && -f "$SKYBETA/host/dds/fastdds_localhost.xml" ]] || die "skybeta checkout not found: set SKYBETA_DIR"

docker info >/dev/null 2>&1 || die "cannot talk to docker (is $(id -un) in the docker group? sudo usermod -aG docker $(id -un), then log in again)"
RUNTIME=(); docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' && RUNTIME=(--runtime nvidia)
mkdir -p "$CACHE/home" "$CACHE/ws/src" "$CACHE/ccache"

ARGS=("${RUNTIME[@]}" --shm-size 1g -u "$(id -u):$(id -g)" -e HOME=/cache/home -e CCACHE_DIR=/cache/ccache
      -v "$REPO:/rig:ro" -v "$PX4:/px4" -v "$SKYBETA:/skybeta:ro" -v "$CACHE:/cache"
      -e ROS_DOMAIN_ID="$DOMAIN" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
      -e FASTRTPS_DEFAULT_PROFILES_FILE=/skybeta/host/dds/fastdds_localhost.xml
      -e FASTDDS_DEFAULT_PROFILES_FILE=/skybeta/host/dds/fastdds_localhost.xml)
for v in SPAWN_POSE BATTERY_SOC CAMERA_X CAMERA_Y CAMERA_Z CAMERA_ROLL CAMERA_PITCH CAMERA_YAW VIO_LATENCY_MS VIO_SIGMA_P \
         VIO_DRIFT_P VIO_DRIFT_YAW VIO_OUTAGE_AT_S VIO_OUTAGE_S DEPTH_NOISE_K DEPTH_DROPOUT; do
  [[ -n "${!v:-}" ]] && ARGS+=(-e "$v=${!v}")
done
if [[ -n "${PX4_PARAMS_DUMP:-}" ]]; then
  dump="$(cd "$(dirname "$PX4_PARAMS_DUMP")" && pwd)/$(basename "$PX4_PARAMS_DUMP")"
  [[ "$dump" == "$REPO"/* ]] || die "PX4_PARAMS_DUMP must be inside this repo (it is mounted at /rig), e.g. sitl/params/drone17.params"
  ARGS+=(-e "PX4_PARAMS_DUMP=/rig/${dump#"$REPO"/}")
fi
SRC='set +u; . /opt/ros/humble/setup.bash; . /opt/px4_ws/install/setup.bash; [ -f /cache/ws/install/setup.bash ] && . /cache/ws/install/setup.bash; set -u'
running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }
TTY=(); [[ -t 0 ]] && TTY=(-it)

build_image() { docker build --build-arg JOBS="$JOBS" -t "$IMAGE" "$HERE"; }
build_px4() {
  local want have
  want="$(git -C "$PX4" rev-parse "$PX4_TAG^{commit}" 2>/dev/null)" || die "$PX4 has no $PX4_TAG tag: git submodule update --init --recursive"
  have="$(git -C "$PX4" rev-parse HEAD)"
  [[ "$want" == "$have" ]] || die "sitl/PX4-Autopilot is at $(git -C "$PX4" describe --tags --always), not $PX4_TAG (the drone's firmware).
       cd sitl/PX4-Autopilot && git checkout $PX4_TAG && git submodule update --init --recursive"
  docker run --rm "${ARGS[@]}" "$IMAGE" bash -c "cd /px4 && make -j$JOBS px4_sitl_default"
}
build_ws() {
  # skybeta's packages are built from the read-only checkout (no --symlink-install: ament_python would write egg-info
  # into the source); policynav is this repo. Rebuild after changing either: run_rig.sh build ws
  docker run --rm "${ARGS[@]}" "$IMAGE" bash -c "$SRC
    cd /cache/ws && rm -rf src && mkdir src
    ln -s /skybeta/src/skybeta_flight_interface /skybeta/src/oakd_vslam_bringup src/ && ln -s /rig src/policynav
    MAKEFLAGS=-j$JOBS colcon build --parallel-workers 2 --packages-select skybeta_flight_interface oakd_vslam_bringup policynav"
}

case "${1:-}" in
  build)
    case "${2:-all}" in
      image) build_image ;; px4) build_px4 ;; ws) build_ws ;;
      all) build_image && build_px4 && build_ws ;;
      *) die "build image|px4|ws|all" ;;
    esac ;;
  up)
    [[ -x "$PX4/build/px4_sitl_default/bin/px4" ]] || die "PX4 SITL not built: run_rig.sh build"
    [[ -f "$CACHE/ws/install/setup.bash" ]] || die "workspace not built: run_rig.sh build ws"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" "${ARGS[@]}" "$IMAGE" /rig/sitl/rig_entry.sh >/dev/null
    echo "[rig] starting (domain $DOMAIN, skybeta $SKYBETA); waiting for READY ..."
    for _ in $(seq 180); do
      running || { docker logs --tail 30 "$NAME"; die "rig container exited"; }
      docker exec "$NAME" grep -q READY /tmp/rig/rig.log 2>/dev/null && { docker exec "$NAME" tail -n 3 /tmp/rig/rig.log; exit 0; }
      docker exec "$NAME" grep -q ERROR /tmp/rig/rig.log 2>/dev/null && { docker exec "$NAME" cat /tmp/rig/rig.log; die "rig failed"; }
      sleep 1
    done
    die "not READY after 180 s: run_rig.sh logs" ;;
  team)
    shift; running || die "rig not up: run_rig.sh up"
    LAUNCH="${TEAM_LAUNCH:-skybeta.launch.py}"
    EXTRA=(); [[ "$LAUNCH" == skybeta.launch.py ]] && EXTRA=(stereo_backend:=none)
    exec docker exec "${TTY[@]}" "$NAME" bash -c "$SRC; exec ros2 launch policynav $LAUNCH ${EXTRA[*]} $*" ;;
  pilot)
    shift; running || die "rig not up: run_rig.sh up"
    exec docker exec "${TTY[@]}" "$NAME" python3 /rig/sitl/pilot.py "$@" ;;
  sh) running || die "rig not up"; exec docker exec -it "$NAME" bash -c "$SRC; cd /tmp/rig; exec bash" ;;
  logs) exec docker exec "$NAME" bash -c 'tail -n 20 /tmp/rig/*.log' ;;
  down) docker rm -f "$NAME" ;;
  *) sed -n 2,22p "$0"; exit 1 ;;
esac
