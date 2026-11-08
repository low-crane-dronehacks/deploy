#!/bin/bash
# Runs INSIDE the rig container (run_rig.sh up): everything the skybeta drone runs below our team launch, with PX4 SITL
# v1.16.2 + Gazebo in place of the Pixhawk, the airframe and the OAK:
#
#   gz sim (headless, our course)       x500_skybeta: forward depth camera on the OAK mount + 4 m down rangefinder
#   PX4 SITL v1.16.2                    airframe 4090 = 4001_gz_x500 + skybeta fleet params (+ drone dump) + SITL overrides
#   MicroXRCEAgent udp4 :8888           skybeta's agent_refs_local.xml (UXRCE_DDS_PTCFG=2), like run_agent.sh
#   gz_vio.py                           truth -> /visual_slam/tracking/odometry          (stands in for cuVSLAM)
#   gz_depth.py                         depth camera -> /policynav/depth/image + TF       (stands in for our stereo)
#   px4_odom_bridge                     skybeta's own bridge, skybeta's launch file       (as run_bridge.sh)
#
# Then `run_rig.sh team` starts our launch (flight_interface + policy), as the stack's competitor window does, and
# `run_rig.sh pilot fly` plays the safety pilot. Logs: /tmp/rig/*.log.
set -u
RIG=/rig/sitl; PX4=/px4; SKY=/skybeta; B=$PX4/build/px4_sitl_default
WORLD=dronehacks; MODEL=x500_skybeta_0; AIRFRAME=4090_gz_x500_skybeta
mkdir -p /tmp/rig; cd /tmp/rig
set +u; . /opt/ros/humble/setup.bash; . /opt/px4_ws/install/setup.bash; . /cache/ws/install/setup.bash; set -u
log() { echo "[rig] $*" | tee -a /tmp/rig/rig.log; }
die() { log "ERROR: $*"; exit 1; }
[[ -x $B/bin/px4 ]] || die "no PX4 build at $B: run_rig.sh build"

# Camera mount: base_link -> camera, FLU, metres/radians -- the CAMERA_* keys of skybeta's drone.env
: "${CAMERA_X:=0.12}" "${CAMERA_Y:=0.0}" "${CAMERA_Z:=0.0}" "${CAMERA_ROLL:=0.0}" "${CAMERA_PITCH:=0.0}" "${CAMERA_YAW:=0.0}"
: "${DEPTH_RATE:=15}" "${DEPTH_NEAR:=0.1}" "${DEPTH_FAR:=20.0}"
export CAMERA_X CAMERA_Y CAMERA_Z CAMERA_ROLL CAMERA_PITCH CAMERA_YAW DEPTH_RATE DEPTH_NEAR DEPTH_FAR

# --- world + model --------------------------------------------------------------------------------------------------
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:-} GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
. $B/rootfs/gz_env.sh                                      # PX4's model / plugin / server-config paths
python3 $RIG/add_normals.py $RIG/world /tmp/rig/world      # gz-physics crashes on normal-less OBJ meshes
mkdir -p /tmp/rig/models
rm -rf /tmp/rig/models/*
cp -r "$PX4_GZ_MODELS/x500" "$PX4_GZ_MODELS/x500_base" /tmp/rig/models/
cp -r $RIG/models/x500_skybeta /tmp/rig/models/
python3 - <<'PYEOF'
import os
src = '/rig/sitl/models/x500_skybeta/model.sdf.in'
s = open(src).read()
for k in ('CAMERA_X', 'CAMERA_Y', 'CAMERA_Z', 'CAMERA_ROLL', 'CAMERA_PITCH', 'CAMERA_YAW', 'DEPTH_RATE', 'DEPTH_NEAR', 'DEPTH_FAR'):
    s = s.replace(f'@{k}@', str(float(os.environ[k])))
assert '@' not in s.split('-->', 1)[1], 'unfilled placeholder in model.sdf.in'
open('/tmp/rig/models/x500_skybeta/model.sdf', 'w').write(s)
PYEOF
rm -f /tmp/rig/models/x500_skybeta/model.sdf.in

# Battery: BATTERY_SOC (0..1, default 1) -> loaded voltage at hover from the 4S 5000 mAh model (battery.py, copied from
# dronehacks-sim) -> thrust factor ~(V/V_ref)^2, applied to the x500 motor constants. Static per flight.
SOC=${BATTERY_SOC:-1.0}
TF=$(cd $RIG && PYTHONPATH=$RIG python3 -c "from battery import LiPoPack as P; p=P.from_json('battery_4s5000.json'); v,i=p.solve($SOC,p.hover_thrust_n); print(round(float(p.thrust_factor(v)),4))")
python3 - "$TF" <<'PYEOF'
import glob, re, sys
f = float(sys.argv[1])
for path in glob.glob('/tmp/rig/models/*/model.sdf'):
    s = open(path).read()
    s2 = re.sub(r'<motorConstant>([0-9.eE+-]+)</motorConstant>', lambda m: f'<motorConstant>{float(m.group(1)) * f:.6e}</motorConstant>', s)
    if s2 != s:
        open(path, 'w').write(s2)
PYEOF
log "battery: SoC $SOC -> hover thrust factor $TF"
export GZ_SIM_RESOURCE_PATH=/tmp/rig/models:/tmp/rig/world:$GZ_SIM_RESOURCE_PATH

# The nvidia runtime may ship the EGL libs without the glvnd vendor file; ogre2 then falls back to mesa and segfaults
# headless. The image carries one (Dockerfile); say so if the runtime replaced it with nothing usable.
[[ -s /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] || log "WARNING: no glvnd EGL vendor file; gz rendering may fail"

# --- 1. Gazebo server ---------------------------------------------------------------------------------------------------
gz sim -r -s --headless-rendering --verbose=1 /tmp/rig/world/$WORLD.sdf > gz.log 2>&1 &
for _ in $(seq 60); do gz topic -l 2>/dev/null | grep -q "^/world/$WORLD/clock" && break; sleep 1; done
gz topic -l 2>/dev/null | grep -q "^/world/$WORLD/clock" || die "gz did not start (gz.log)"

# Spawn ourselves with the full launch pose: px4-rc.gzsim (v1.16.2) only applies x y z of PX4_GZ_MODEL_POSE, not yaw.
POSE="${SPAWN_POSE:-$(grep -v '^#' $RIG/world/spawn.txt | head -1)}"      # x,y,z,roll,pitch,yaw (ENU)
IFS=, read -r PX PY PZ _PR _PP PYAW <<< "$POSE"
QZ=$(python3 -c "import math;print(math.sin(${PYAW:-0}/2))"); QW=$(python3 -c "import math;print(math.cos(${PYAW:-0}/2))")
gz service -s /world/$WORLD/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 10000 \
  --req "sdf_filename: \"/tmp/rig/models/x500_skybeta/model.sdf\", name: \"$MODEL\", allow_renaming: false, pose: {position: {x: $PX, y: $PY, z: $PZ}, orientation: {z: $QZ, w: $QW}}" \
  > spawn.log 2>&1 || die "model spawn failed (spawn.log)"
log "spawned $MODEL at $POSE"

# --- 2. PX4 airframe + params ------------------------------------------------------------------------------------------
python3 $RIG/params.py --fleet $SKY/px4/params ${PX4_PARAMS_DUMP:+--dump "$PX4_PARAMS_DUMP"} \
  --overrides $RIG/px4_params.txt --out $B/etc/init.d-posix/airframes/$AIRFRAME | tee -a rig.log || die "params.py failed"
rm -f $B/rootfs/parameters*.bson $B/rootfs/dataman        # fresh params every start: the airframe defaults are the truth

# --- 3. agent + PX4 ------------------------------------------------------------------------------------------------------
# Like skybeta's run_agent.sh in udp4 mode: PX4 asks for its participant by reference (PTCFG=2), built from the refs file.
MicroXRCEAgent udp4 -p 8888 -r $SKY/host/dds/agent_refs_local.xml > agent.log 2>&1 &
export PX4_SYS_AUTOSTART=${AIRFRAME%%_*} PX4_GZ_MODEL_NAME=$MODEL PX4_GZ_WORLD=$WORLD HEADLESS=1
# rcS copies ROS_DOMAIN_ID into UXRCE_DDS_DOM_ID, so PX4 joins the rig's domain.
(cd $B/rootfs && exec $B/bin/px4 -d $B/etc > /tmp/rig/px4.log 2>&1) &
for _ in $(seq 60); do grep -q "Ready for takeoff\|INFO  \[commander\] Ready" px4.log 2>/dev/null && break; sleep 1; done
grep -q "uxrce_dds_client\|UXRCE" px4.log || log "WARNING: no uxrce_dds_client line in px4.log yet"

# --- 4. ROS side: the sim feeds + skybeta's bridge -----------------------------------------------------------------------
python3 $RIG/gz_vio.py --ros-args -p model:=$MODEL -p gz_topic:=/world/$WORLD/dynamic_pose/info \
  -p latency_ms:=${VIO_LATENCY_MS:-65} -p sigma_p:=${VIO_SIGMA_P:-0.005} -p drift_p:=${VIO_DRIFT_P:-0.002} \
  -p drift_yaw:=${VIO_DRIFT_YAW:-0.0005} -p outage_at_s:=${VIO_OUTAGE_AT_S:--1.0} -p outage_s:=${VIO_OUTAGE_S:-0.0} \
  > vio.log 2>&1 &
python3 $RIG/gz_depth.py --ros-args -p camera_x:=$CAMERA_X -p camera_y:=$CAMERA_Y -p camera_z:=$CAMERA_Z \
  -p camera_roll:=$CAMERA_ROLL -p camera_pitch:=$CAMERA_PITCH -p camera_yaw:=$CAMERA_YAW \
  -p noise_k:=${DEPTH_NOISE_K:-0.0} -p dropout:=${DEPTH_DROPOUT:-0.0} > depth.log 2>&1 &
ros2 launch oakd_vslam_bringup px4_bridge.launch.py > bridge.log 2>&1 &

# ready = PX4 is publishing to ROS and the bridge has something to forward
for _ in $(seq 60); do
  timeout 3 ros2 topic echo --once --qos-reliability best_effort /fmu/out/vehicle_status_v1 >/dev/null 2>&1 \
    || timeout 3 ros2 topic echo --once --qos-reliability best_effort /fmu/out/vehicle_status >/dev/null 2>&1 \
    && break
  sleep 1
done
log "READY: domain $ROS_DOMAIN_ID; run_rig.sh team, then run_rig.sh pilot fly (--arm to fly). Logs in /tmp/rig"
wait
