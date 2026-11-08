# Organizers' simulator (FlyX) on our own machine

FlyX is the organizers' simulator: Isaac Sim 5.1 + Pegasus 5.1 + PX4 SITL + ROS 2 Jazzy + cuVSLAM. It is a
**held-out sim2sim test**. Evaluate in it, but never tune policies or this stack to it.

**License:** FlyX is proprietary. The organizers allowed local, non-distributed use; the written record is
kept by Colin. Never commit, push or share the FlyX source or images. Only these instructions live here.

## Where it runs
**pear-omen** (RTX 5090, 32 GB, driver 590, Ubuntu 22.04 host):
- Docker container `flyx`: ubuntu:24.04 with ROS 2 Jazzy, `--init --network host`.
- `docker.sock` is mounted so the two cuVSLAM images run as sibling containers.
- Everything lives in the docker volume `flyx_home` (`/var/lib/docker/volumes/flyx_home/_data`).
- Deviations from their pins (22.04 host, cupy 13.6 + numpy 1.26 in Isaac's python) are listed in `NOTES.md`
  in that volume.

A second lab PC cannot run it: driver 550 < 580, and their native tracker is built for RTX 50-series (SM120).

## Host prerequisite
FastDDS needs 16 MB UDP buffers, or sensor frames are dropped and the sim falls to ~0.1x real time. Run this
once per boot (root):
```bash
sudo sysctl -w net.core.rmem_max=33554432 net.core.wmem_max=33554432
```
To make it persistent, put both settings in `/etc/sysctl.d/`.

## Start / status / stop
```bash
P=/var/lib/docker/volumes/flyx_home/_data
docker exec -d flyx bash -c "cd $P/flyx-sim/flyx-sim && python3 -m runtime start \
  --scenario configs/gps_denied/dronehacks_course_v3.config.json --session s1 --wait > $P/start.log 2>&1"
docker exec -w $P/flyx-sim/flyx-sim flyx python3 -m runtime status --session s1
docker exec -w $P/flyx-sim/flyx-sim flyx python3 -m runtime close  --session s1
```
- If the container is recreated, keep `--init`: a zombie XRCE agent from a failed run blocks the next start.
- With the sim up it uses about 7 GB of VRAM, so it shares the card with a training job.

Then run the policy against it with `ros2 launch launch/policynav.launch.py hardware:=flyx` (dry run by default).
