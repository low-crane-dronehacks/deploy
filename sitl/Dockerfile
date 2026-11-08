# PX4 SITL rig matching the skybeta drone: ROS 2 Humble, PX4 v1.16.2 (the sitl/PX4-Autopilot submodule, bind-mounted and
# built by `run_rig.sh build`), Gazebo Harmonic, Micro-XRCE-DDS-Agent v2.4.3 and px4_msgs release/1.16 (what skybeta pins).
# Multi-arch: builds on the Jetson (arm64) and on an x86 lab PC.
#
# The image holds only the toolchain. Built at `run_rig.sh build` time, into host directories:
#   sitl/PX4-Autopilot/build/px4_sitl_default    PX4 SITL
#   sitl/.cache/ws                               skybeta's flight_interface + px4_odom_bridge (from the read-only skybeta
#                                                checkout) and policynav (this repo), overlaid on px4_msgs
FROM ros:humble-ros-base-jammy
ARG JOBS=4
ARG XRCE_AGENT_TAG=v2.4.3
ARG PX4_MSGS_BRANCH=release/1.16
ENV DEBIAN_FRONTEND=noninteractive

# Gazebo Harmonic (OSRF repo; PX4 v1.16 requires gz-sim >= 8) + its Python bindings, which the rig's gz <-> ROS bridges use
# (ros_gz for Humble + Harmonic is not packaged for arm64).
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release ca-certificates \
 && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      gz-harmonic python3-gz-transport13 python3-gz-msgs10 libunwind-dev \
      git cmake ninja-build build-essential ccache python3-pip python3-dev python3-venv xsltproc libxml2-utils \
      libeigen3-dev zlib1g-dev python3-numpy python3-opencv libopencv-dev bc procps iproute2 gdb \
      python3-colcon-common-extensions ros-humble-tf2-ros ros-humble-launch-ros ros-humble-nav-msgs \
      ros-humble-sensor-msgs ros-humble-std-srvs ros-humble-geometry-msgs \
 && rm -rf /var/lib/apt/lists/*

# PX4's Python build/runtime requirements, from the pinned tree (pymavlink included: sitl/pilot.py uses it)
# numpy < 2: apt's python3-opencv (and rclpy's numpy users) are built against numpy 1.x; pandas/matplotlib would pull 2.x.
COPY PX4-Autopilot/Tools/setup/requirements.txt /tmp/px4-requirements.txt
RUN echo 'numpy<2' > /tmp/constraints.txt && pip3 install --no-cache-dir -c /tmp/constraints.txt -r /tmp/px4-requirements.txt

# The nvidia container runtime ships the EGL libs but not always the glvnd vendor file: without it ogre2 falls back to
# mesa and segfaults in headless rendering (seen with snap Docker on x86). A runtime that mounts its own file wins.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] \
 || echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Micro XRCE-DDS agent, same tag as skybeta's image (Dockerfile.jetson: XRCE_AGENT_TAG=v2.4.3)
RUN git clone --depth 1 --branch ${XRCE_AGENT_TAG} https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /opt/xrce \
 && cmake -S /opt/xrce -B /opt/xrce/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /opt/xrce/build -j${JOBS} && cmake --install /opt/xrce/build && ldconfig && rm -rf /opt/xrce

# px4_msgs at the branch matching PX4 v1.16 (no *_v1 topic suffixes for the messages we use)
RUN mkdir -p /opt/px4_ws/src \
 && git clone --depth 1 --branch ${PX4_MSGS_BRANCH} https://github.com/PX4/px4_msgs.git /opt/px4_ws/src/px4_msgs \
 && . /opt/ros/humble/setup.sh && cd /opt/px4_ws && MAKEFLAGS=-j${JOBS} colcon build --parallel-workers 1 \
 && rm -rf /opt/px4_ws/build /opt/px4_ws/log

ENV NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all
WORKDIR /rig
