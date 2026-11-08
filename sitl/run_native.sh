#!/bin/bash
# Native PX4 SITL + Gazebo Harmonic on this machine (no Docker), with the Gazebo window. The container rig is
# sitl/run_rig.sh + rig_entry.sh; this is the same stack with your own paths (docs/SITL.md, "Native").
#
#   sitl/run_native.sh up        start: gz server (+ GUI) with our world, x500_stereo at the launch mark, PX4, agent,
#                                ROS bridges, and PX4's position source (EV_SOURCE). Returns when it is up.
#   sitl/run_native.sh status    what is running, and PX4's local-position validity
#   sitl/run_native.sh logs      tail every log
#   sitl/run_native.sh down      stop exactly what `up` started (recorded process groups), nothing else
#
# Options (environment):
#   EV_SOURCE=truth   (default) Gazebo's true pose + noise -> PX4 external vision (sitl/ev_bridge.py), as the rig
#   EV_SOURCE=cuvslam stereo + IMU from the model -> cuVSLAM (onboard/slam, venv) -> onboard/vio/ev_bridge.py -> PX4,
#                     the real drone's chain.
#                     Gazebo's truth is still published, on /sitl/truth_odometry (px4_msgs/VehicleOdometry, NED).
#                     cuVSLAM runs Inertial (stereo + IMU, vio.yaml) unless VSLAM_IMU=0; SKYBETA=1 defaults to
#                     VSLAM_IMU=0, stereo only, as skybeta runs it on the drone ("IMU fusion off", SKYBETA.md).
#   EV_SOURCE=truth_odom  Gazebo's truth published AS /visual_slam/tracking/odometry (sitl/truth_odom.py, cuVSLAM's
#                     convention) -> the same onboard bridge -> PX4: the real chain with a perfect odometry source,
#                     to separate vision problems from everything else.
#                     Heading: the bridge is given the spawn yaw as its surveyed reference (heading_enu_rad, as a
#                     deployment's onboard.yaml would), so PX4 gets NED and a usable heading; EV_HEADING=none keeps
#                     the unreferenced FRD frame (PX4 then fuses it but reports heading_good_for_control = false).
#   SKYBETA=1         the real drone's start procedure: PX4 takes simulated RC (COM_RC_IN_MODE 0),
#                     sitl/flight_interface_sim.py stands in for skybeta's flight_interface (twist/pose -> Offboard,
#                     Offboard requested once when armed in Position), the bridge reads cuVSLAM's odom frame as the
#                     world (EV_HEADING=odom). Needs EV_SOURCE=cuvslam or truth_odom. Then: the team node (e.g.
#                     climb_forward_hold_skybeta.py fly:=true) + python3 sitl/sim_pilot.py --arm (arms; approval).
#   GUI=1             (default) open the Gazebo window; GUI=0 = headless
#   TEXTURES=1        (default) textured course (sitl/texture_world.py: orange walls, blue window, grey carpet,
#                     see-through net, stone boulders); TEXTURES=0 = the flat grey world. Collision is the same.
#   PX4_LOG_DIR       where PX4's .ulg flight logs go (default ~/dronehacks_px4_logs/<date>/): kept across runs; the
#                     run dir (RUN_DIR, default /tmp/dronehacks-native) is wiped on every 'up'.
#   MODEL=dronehacks_stereo (default): OUR quad (sitl/models/make_dronehacks_quad.py, from the training plant:
#                     1.5 kg, rotors +-0.13 m, 10 N each, ~0.25 m reach) + OAK depth + stereo + IMU at the nose;
#                     PX4 gets the matching mixer (CA_ROTOR*), hover thrust and scaled rate gains below
#   MODEL=dronehacks_quad (no cameras) | x500_stereo | x500_depth (PX4's x500: ~0.39 m reach, too big for the
#                     course's tight passes, see docs/SITL.md)
#   SPAWN=launch      (default) the launch mark (world/spawn.txt), as the competition: only 0.39 m from the wall behind
#   SPAWN=open        on the landing pad, NED (-1.15, -1.95), facing north: 1.05 m clearance all round
#   SPAWN_POSE=x,y,z,roll,pitch,yaw  (ENU) overrides both
#   BATTERY_SOC=0..1  pack charge -> thrust scaling (as the rig)
#   ROS_DOMAIN_ID     honoured end to end (PX4's UXRCE_DDS_DOM_ID is set to it)
#   PX4_DIR=~/PX4-v1.17  PX4_MSGS_WS=~/isam2_ws  XRCE_AGENT=~/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent
#   CUVSLAM_VENV=~/cuvslam-venv  RUN_DIR=/tmp/dronehacks-native
#
# Starting the simulator never arms. Arming (e.g. climb_forward_hold.py arm:=true) needs the team's approval for that
# flight when SITL runs on a shared machine (AGENTS.md).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
PX4_DIR=${PX4_DIR:-$HOME/PX4-v1.17}; B=$PX4_DIR/build/px4_sitl_default
WS=${PX4_MSGS_WS:-$HOME/isam2_ws}; AGENT=${XRCE_AGENT:-$HOME/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent}
VENV=${CUVSLAM_VENV:-$HOME/cuvslam-venv}; RUN=${RUN_DIR:-/tmp/dronehacks-native}; PX4_LOGS=${PX4_LOG_DIR:-$HOME/dronehacks_px4_logs}
EV_SOURCE=${EV_SOURCE:-truth}; GUI=${GUI:-1}; TEXTURES=${TEXTURES:-1}; MODEL=${MODEL:-dronehacks_stereo}; WORLD=dronehacks; NAME=${MODEL}_0
PIDS=$RUN/pids

env_setup() {   # clean ROS environment: no conda, ROS + px4_msgs workspace, our + PX4's Gazebo resources
  export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -e anaconda -e miniconda | paste -sd:)
  unset CONDA_PREFIX PYTHONPATH
  set +u; . /opt/ros/jazzy/setup.bash; . "$WS/install/setup.bash"; set -u
  export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:-} GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
  . "$B/rootfs/gz_env.sh"
  export GZ_SIM_RESOURCE_PATH=$RUN/models:$RUN/world:$GZ_SIM_RESOURCE_PATH   # private model copies shadow PX4's
  # Gazebo transport is machine-wide: a partition per run keeps two simulators from hearing each other.
  # (`gz topic` etc. from another terminal: export GZ_PARTITION=$(cat $RUN/gz_partition))
  export GZ_PARTITION=${GZ_PARTITION:-dronehacks_native_$(echo "$RUN" | md5sum | cut -c1-8)}
}

start() {       # start <name> <command...>: own session (so `down` can stop the whole group), log to $RUN/<name>.log
  local name=$1; shift
  setsid "$@" > "$RUN/$name.log" 2>&1 < /dev/null &
  echo "$! $name" >> "$PIDS"
}

wait_for() {    # wait_for <seconds> <description> <test command...>
  local n=$1 what=$2; shift 2
  for _ in $(seq "$n"); do "$@" > /dev/null 2>&1 && return 0; sleep 1; done
  echo "timed out waiting for $what (logs: $RUN)"; return 1
}

up() {
  [ -s "$PIDS" ] && { echo "already up (pids in $PIDS); run '$0 down' first"; exit 1; }
  case "$EV_SOURCE" in truth|cuvslam|truth_odom) ;; *) echo "EV_SOURCE must be truth | cuvslam | truth_odom"; exit 1;; esac
  # per model: stereo pair?, depth camera optical frame in base_link (x y z), stereo link in base_link (x z)
  local STEREO=0 DEPTH_TF="" STEREO_TF=""
  case "$MODEL" in
    dronehacks_stereo) STEREO=1; DEPTH_TF="0.13233 0.0 0.01878"; STEREO_TF="0.12 0.02" ;;
    dronehacks_quad)   ;;
    x500_stereo)       STEREO=1; DEPTH_TF="0.13233 0.0 0.02078"; STEREO_TF="0.13 0.20" ;;
    x500_depth)        DEPTH_TF="0.13233 0.0 0.02078" ;;
    *) echo "MODEL must be dronehacks_stereo | dronehacks_quad | x500_stereo | x500_depth"; exit 1 ;;
  esac
  [ "$EV_SOURCE" != truth ] && [ "$STEREO" = 0 ] && [ "$EV_SOURCE" = cuvslam ] && { echo "EV_SOURCE=cuvslam needs a stereo model (dronehacks_stereo / x500_stereo)"; exit 1; }
  SKYBETA=${SKYBETA:-0}
  [ "$SKYBETA" = 1 ] && [ "$EV_SOURCE" = truth ] && {
    echo "SKYBETA=1 needs EV_SOURCE=cuvslam or truth_odom (skybeta nodes read /visual_slam/tracking/odometry)"; exit 1; }
  [ "$SKYBETA" = 1 ] && EV_HEADING=${EV_HEADING:-odom}
  VSLAM_IMU=${VSLAM_IMU:-$([ "$SKYBETA" = 1 ] && echo 0 || echo 1)}
  for f in "$B/bin/px4" "$AGENT" "$WS/install/setup.bash"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
  [ "$EV_SOURCE" = cuvslam ] && { [ -x "$VENV/bin/python" ] || { echo "missing $VENV (docs/CUVSLAM_LOCAL.md)"; exit 1; }; }
  # One PX4 SITL per machine here (instance 0, agent port 8888): refuse instead of half-starting next to another one.
  local BUSY='^[^ ]*px4_sitl_default/bin/px4 |^[^ ]*MicroXRCEAgent udp4|^gz sim '   # anchored: not shells mentioning them
  if pgrep -f "$BUSY" > /dev/null; then
    echo "another PX4 SITL / XRCE agent / Gazebo is already running on this machine:"
    pgrep -af "$BUSY" | sed 's/^/  /'
    echo "stop it first (its own 'run_native.sh down', with its RUN_DIR)"; exit 1
  fi
  rm -rf "$RUN"; mkdir -p "$RUN/models" "$RUN/rootfs"; : > "$PIDS"
  # PX4 flight logs outlive the run dir (wiped above on every up): rootfs/log -> $PX4_LOGS/<date>/*.ulg
  mkdir -p "$PX4_LOGS"; ln -s "$PX4_LOGS" "$RUN/rootfs/log"
  env_setup
  echo "$GZ_PARTITION" > "$RUN/gz_partition"
  local DOMAIN=${ROS_DOMAIN_ID:-0}
  echo "native SITL: model $MODEL, EV_SOURCE=$EV_SOURCE, SKYBETA=$SKYBETA, VSLAM_IMU=$VSLAM_IMU, spawn ${SPAWN_POSE:-${SPAWN:-launch}}, GUI=$GUI, ROS_DOMAIN_ID=$DOMAIN, logs in $RUN, PX4 flight logs in $PX4_LOGS"

  # 1. world (normals added: gz-physics crashes on normal-less OBJ meshes) + private model copies
  python3 "$HERE/add_normals.py" "$HERE/world" "$RUN/world" > "$RUN/world.log" 2>&1
  if [ "$TEXTURES" = 1 ]; then   # textured visual meshes (walls, window, carpet, net, boulders); collision unchanged
    python3 "$HERE/texture_world.py" "$RUN/world" >> "$RUN/world.log" 2>&1 || { echo "texture_world.py failed: $RUN/world.log"; exit 1; }
  fi
  cp -r "$PX4_GZ_MODELS/x500" "$PX4_GZ_MODELS/x500_depth" "$RUN/models/"
  [ -d "$PX4_GZ_MODELS/x500_base" ] && cp -r "$PX4_GZ_MODELS/x500_base" "$RUN/models/"
  cp -r "$HERE/models/x500_stereo" "$HERE/models/dronehacks_quad" "$HERE/models/dronehacks_stereo" "$RUN/models/"
  local SOC=${BATTERY_SOC:-1.0} TF
  TF=$(cd "$HERE" && PYTHONPATH="$HERE" python3 -c "from battery import LiPoPack as P; p=P.from_json('battery_4s5000.json'); v,i=p.solve($SOC,p.hover_thrust_n); print(round(float(p.thrust_factor(v)),4))")
  python3 - "$TF" "$RUN/models" <<'PYEOF' > "$RUN/battery.log"
import re, sys, glob
f = float(sys.argv[1])
for path in glob.glob(sys.argv[2] + '/*/model.sdf'):
    s = open(path).read()
    s2 = re.sub(r'<motorConstant>([0-9.eE+-]+)</motorConstant>', lambda m: f'<motorConstant>{float(m.group(1)) * f:.6e}</motorConstant>', s)
    if s2 != s:
        open(path, 'w').write(s2)
        print(f'battery: scaled motorConstant x{f} in {path}')
PYEOF
  echo "battery: SoC $SOC -> hover thrust factor $TF" >> "$RUN/battery.log"

  # 2. gazebo server (EGL headless rendering for the sensors), then the window if GUI=1
  start gz_server gz sim -r -s --headless-rendering -v 1 "$RUN/world/$WORLD.sdf"
  wait_for 60 "the Gazebo server" bash -c "gz topic -l | grep -q '^/world/$WORLD/clock'" || { down; exit 1; }
  [ "$GUI" = 1 ] && start gz_gui gz sim -g -v 1

  # 3. spawn the model ourselves with the full launch pose (PX4 1.17 applies only x y z of PX4_GZ_MODEL_POSE)
  local DEFAULT_POSE PX PY PZ PR PP PYAW QZ QW
  case "${SPAWN:-launch}" in
    open)   DEFAULT_POSE="-1.95,-1.15,0.05,0,0,1.5708" ;;   # landing pad, NED (-1.15,-1.95): 1.05 m clearance, faces north
    launch) DEFAULT_POSE=$(grep -v '^#' "$HERE/world/spawn.txt" | head -1) ;;
    *) echo "SPAWN must be open | launch (or set SPAWN_POSE)"; down; exit 1 ;;
  esac
  local POSE=${SPAWN_POSE:-$DEFAULT_POSE}
  IFS=, read -r PX PY PZ PR PP PYAW <<< "$POSE"
  QZ=$(python3 -c "import math;print(math.sin(${PYAW:-0}/2))"); QW=$(python3 -c "import math;print(math.cos(${PYAW:-0}/2))")
  gz service -s /world/$WORLD/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 10000 \
    --req "sdf_filename: \"$RUN/models/$MODEL/model.sdf\", name: \"$NAME\", allow_renaming: false, pose: {position: {x: $PX, y: $PY, z: $PZ}, orientation: {z: $QZ, w: $QW}}" \
    > "$RUN/spawn.log" 2>&1
  grep -q "data: true" "$RUN/spawn.log" || { echo "spawn failed: $(cat "$RUN/spawn.log")"; down; exit 1; }

  # 4. PX4: private etc/ copy with the indoor / external-vision params (~/PX4-v1.17 stays untouched), fresh rootfs
  cp -r "$B/etc" "$RUN/etc"
  { echo; echo "# sitl/run_native.sh"; cat "$HERE/px4_params.txt"; echo "param set-default UXRCE_DDS_DOM_ID $DOMAIN"
    [ "$SKYBETA" = 1 ] && echo "param set-default COM_RC_IN_MODE 0"
    case "$MODEL" in dronehacks_*)
      # our airframe (make_dronehacks_quad.py): mixer at the real rotor positions (FRD, x500 rotor order/turning),
      # hover thrust ~0.50 (10 N rotors, 1.5 kg, ESC linear in speed), rate gains scaled for its ~1.6x roll/pitch and
      # ~2.3x yaw acceleration per command vs the x500 (less arm, much less inertia). A STARTING tune: check a hover.
      echo "param set-default CA_ROTOR0_PX 0.13";  echo "param set-default CA_ROTOR0_PY 0.13";  echo "param set-default CA_ROTOR0_KM 0.05"
      echo "param set-default CA_ROTOR1_PX -0.13"; echo "param set-default CA_ROTOR1_PY -0.13"; echo "param set-default CA_ROTOR1_KM 0.05"
      echo "param set-default CA_ROTOR2_PX 0.13";  echo "param set-default CA_ROTOR2_PY -0.13"; echo "param set-default CA_ROTOR2_KM -0.05"
      echo "param set-default CA_ROTOR3_PX -0.13"; echo "param set-default CA_ROTOR3_PY 0.13";  echo "param set-default CA_ROTOR3_KM -0.05"
      echo "param set-default MPC_THR_HOVER 0.50"
      echo "param set-default MC_ROLLRATE_P 0.095";  echo "param set-default MC_ROLLRATE_I 0.127";  echo "param set-default MC_ROLLRATE_D 0.0019"
      echo "param set-default MC_PITCHRATE_P 0.095"; echo "param set-default MC_PITCHRATE_I 0.127"; echo "param set-default MC_PITCHRATE_D 0.0019"
      echo "param set-default MC_YAWRATE_P 0.085";   echo "param set-default MC_YAWRATE_I 0.043" ;;
    esac; } \
    >> "$RUN/etc/init.d-posix/airframes/4001_gz_x500"
  start agent "$AGENT" udp4 -p 8888
  start px4 env -C "$RUN/rootfs" PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=gz_x500_depth PX4_GZ_MODEL_NAME="$NAME" \
    PX4_GZ_WORLD=$WORLD HEADLESS=1 "$B/bin/px4" -d "$RUN/etc"
  wait_for 60 "PX4 on ROS" bash -c "ros2 topic list | grep -q /fmu/out/vehicle_status" || { down; exit 1; }

  # 5. ROS side: /clock, depth (+ stereo + IMU), TF, depth relay
  local DEPTH INFO BR=()
  DEPTH=$(gz topic -l | grep -m1 -E "depth_camera$"); INFO=$(gz topic -l | grep -m1 -E "depth_camera/camera_info$|^/camera_info$")
  BR+=("/world/$WORLD/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock")
  local REMAP=(-r "/world/$WORLD/clock:=/clock")
  if [ -n "$DEPTH_TF" ] && [ -n "$DEPTH" ]; then
    BR+=("$DEPTH@sensor_msgs/msg/Image[gz.msgs.Image" "$INFO@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo")
    REMAP+=(-r "$DEPTH:=/camera/depth/image_raw_gz" -r "$INFO:=/camera/depth/camera_info")
  fi
  if [ "$STEREO" = 1 ]; then
    BR+=("/stereo/left/image_gz@sensor_msgs/msg/Image[gz.msgs.Image" "/stereo/right/image_gz@sensor_msgs/msg/Image[gz.msgs.Image"
         "/stereo/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
         "/stereo/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
         "/stereo/imu_gz@sensor_msgs/msg/Imu[gz.msgs.IMU")
    REMAP+=(-r /stereo/left/image_gz:=/stereo/left/image -r /stereo/right/image_gz:=/stereo/right/image -r /stereo/imu_gz:=/stereo/imu)
  fi
  start bridge ros2 run ros_gz_bridge parameter_bridge "${BR[@]}" --ros-args "${REMAP[@]}"
  # camera frames in base_link (per model, see the case above). The OakD-Lite sensor sits (.01233, -.03, .01878) from
  # its model origin, which the models put at base_link (x, 0.03, ~0): base_link is already at the landing-gear
  # height, so the x500 depth frame is ~0.02 m above base_link (the rig's 0.26 m counted the gear height twice).
  if [ -n "$DEPTH_TF" ]; then
    read -r DX DY DZ <<< "$DEPTH_TF"
    start tf_depth ros2 run tf2_ros static_transform_publisher --x $DX --y $DY --z $DZ \
      --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 --frame-id base_link --child-frame-id camera_optical_frame
  fi
  if [ "$STEREO" = 1 ]; then             # stereo link at (SX, 0, SZ) in base_link, eyes at y = +-0.0375
    read -r SX SZ <<< "$STEREO_TF"
    start tf_left ros2 run tf2_ros static_transform_publisher --x $SX --y 0.0375 --z $SZ \
      --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 --frame-id base_link --child-frame-id oak_left
    start tf_right ros2 run tf2_ros static_transform_publisher --x $SX --y -0.0375 --z $SZ \
      --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 --frame-id base_link --child-frame-id oak_right
    start tf_imu ros2 run tf2_ros static_transform_publisher --x $SX --y 0.0 --z $SZ \
      --frame-id base_link --child-frame-id oak_imu
  fi
  [ -n "$DEPTH_TF" ] && start depth_relay python3 "$HERE/depth_relay.py" --ros-args -p use_sim_time:=true

  # 6. PX4's position source. Exactly one publisher of /fmu/in/vehicle_visual_odometry.
  if [ "$EV_SOURCE" = truth ]; then
    start ev_truth python3 "$HERE/ev_bridge.py" --ros-args -p model:=$NAME -p use_sim_time:=false
  else
    # odometry on /visual_slam/tracking/odometry: cuVSLAM, or Gazebo's truth in cuVSLAM's convention
    if [ "$EV_SOURCE" = cuvslam ]; then
      local NV=$VENV/lib/python3.12/site-packages/nvidia IMU=true
      [ "$VSLAM_IMU" = 0 ] && IMU=false
      start cuvslam env -C "$REPO" LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}" \
        "$VENV/bin/python" -m onboard.slam.cuvslam_lib_node --ros-args --params-file "$REPO/onboard/config/vio.yaml" \
        -p use_sim_time:=true -p imu:=$IMU -p imu_frequency:=200.0
    else
      start truth_odom python3 "$HERE/truth_odom.py" --ros-args -p model:=$NAME -p use_sim_time:=true
    fi
    local HEAD=()
    case "${EV_HEADING:-spawn}" in
      spawn) HEAD=(-p "heading_enu_rad:=${PYAW:-0}"
                   -p "heading_reference:='SITL spawn pose (SPAWN / SPAWN_POSE), yaw ${PYAW:-0} rad ENU'") ;;
      odom)  HEAD=(-p "heading_enu_rad:=0.0"     # skybeta convention: cuVSLAM's odom frame IS the (ENU) world
                   -p "heading_reference:='skybeta convention - cuVSLAM odom frame read as ENU (SKYBETA=1)'") ;;
    esac   # values are YAML: quoted, since ': ' inside would make rcl reject the arguments (the bridge would not start)
    start ev_vslam env -C "$REPO" python3 -m onboard.vio.ev_bridge --ros-args \
      --params-file "$REPO/onboard/config/vision_bridge.yaml" -p use_sim_time:=true "${HEAD[@]}"
    # Gazebo's truth for scoring, NOT to PX4 (same reader as the truth source, remapped away from /fmu/in)
    start ev_truth_log python3 "$HERE/ev_bridge.py" --ros-args -p model:=$NAME -p use_sim_time:=false \
      -r /fmu/in/vehicle_visual_odometry:=/sitl/truth_odometry -r /estimator/healthy:=/sitl/truth_healthy
  fi
  [ "$SKYBETA" = 1 ] && start flight_interface python3 "$HERE/flight_interface_sim.py" --ros-args -p use_sim_time:=false
  echo "up. '$0 status' for PX4's position validity; '$0 down' to stop."
}

down() {
  [ -s "$PIDS" ] || { echo "nothing recorded in $PIDS"; return 0; }
  tac "$PIDS" | while read -r pid name; do kill -INT -- "-$pid" 2> /dev/null && echo "stopping $name"; done
  sleep 3
  tac "$PIDS" | while read -r pid name; do kill -KILL -- "-$pid" 2> /dev/null; done
  : > "$PIDS"; echo "down"
}

status() {
  [ -s "$PIDS" ] || { echo "not running"; return; }
  while read -r pid name; do
    if kill -0 "$pid" 2> /dev/null; then echo "  up    $name ($pid)"; else echo "  DEAD  $name ($pid): see $RUN/$name.log"; fi
  done < "$PIDS"
  env_setup
  timeout 5 ros2 topic echo --once /fmu/out/vehicle_local_position_v1 2> /dev/null \
    | grep -E "^(x|y|z|xy_valid|z_valid|heading_good_for_control):" | tr '\n' ' '; echo
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  logs) tail -n 15 "$RUN"/*.log ;;
  *) sed -n 2,25p "$0"; exit 1 ;;
esac
