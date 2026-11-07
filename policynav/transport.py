"""policynav.transport: subscription QoS shared by the policy nodes (no policy logic here).

LATEST           best effort, keep only the newest sample. For streams where a newer sample supersedes an older one
                 (depth images, odometry): after an executor stall the node reads the newest frame instead of
                 working through up to 5 (SensorDataQoS) or 10 (depth=10, reliable) stale ones. Best effort also
                 means the publisher (cuVSLAM's odometry writer, in the flight-critical VIO process) does not keep
                 acknack/retransmit state for our slow Python reader. Compatible with reliable and best-effort
                 writers alike.
"""
from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

LATEST = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST, depth=1)
