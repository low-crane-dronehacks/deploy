# PX4 policy inputs implementation plan

**Goal:** Replace direct cuVSLAM state in the Skybeta flying policy with PX4 fused velocity and attitude,
retaining the existing estimated downward-distance input and removing altitude fallback.

**Agreed scope:** PX4 1.16 unsuffixed topics; preserve the organizer command interface, stereo depth,
policy scales and forward-launch behavior. Keep range semantics; do not substitute fused altitude.
No flight or arming. Handheld and upstream EKF2 vision fusion remain separate dependencies.

**Implementation:** A ROS-independent `Px4Feedback` validates local-position and attitude messages,
tracks individual sample freshness, and latches in-flight input faults. `skybeta_node.py` consumes it
for observations and command heading; the launch exposes the two topics and watchdog thresholds.

- [x] Establish offline baseline: 90 tests pass using system Python.
- [x] Add failing PX4 input tests, then implement frame-preserving validation and freshness checks.
- [x] Wire the policy node to PX4 and remove altitude fallback; validate interface state freshness.
- [x] Test actual node decision paths and observation/action conversion without ROS middleware.
- [x] Update launch arguments and policy/Skybeta documentation.
- [x] Review diff independently and run final offline checks: 122 tests pass; Python compilation
  and diff whitespace checks pass. Live ROS delivery and interface JSON schema/rate require bench verification.

Review focus: NED/FRD versus ENU signs, duplicate samples, invalid range after a good sample,
reset events between policy ticks, and permanent yielding after pilot takeover/state loss.
