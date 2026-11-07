"""policynav.transport: subscription QoS and TF plumbing shared by the policy nodes (no policy logic here).

LATEST           best effort, keep only the newest sample. For streams where a newer sample supersedes an older one
                 (depth images, odometry): after an executor stall the node reads the newest frame instead of
                 working through up to 5 (SensorDataQoS) or 10 (depth=10, reliable) stale ones. Best effort also
                 means the publisher (cuVSLAM's odometry writer, in the flight-critical VIO process) does not keep
                 acknack/retransmit state for our slow Python reader. Compatible with reliable and best-effort
                 writers alike.

StaticTfListener /tf_static only. tf2_ros.TransformListener (Humble) also subscribes to /tf with RELIABLE keep_last 100,
                 which wakes the Python executor for every dynamic transform cuVSLAM publishes (odom->base_link,
                 map->odom at 30 Hz) and can queue up to 100 of them behind a stall. Our nodes only look up
                 base_link <- <depth frame>, and every link of that chain is static (skybeta's base_link ->
                 oak_parent_frame static_transform_publisher, the depthai driver's calibration TF via a
                 StaticTransformBroadcaster, and our own <left optical> -> _rect static broadcast).
"""
from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

LATEST = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST, depth=1)

# the same QoS tf2_ros uses for /tf_static (transient local, so late joiners get every static transform)
TF_STATIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                           history=HistoryPolicy.KEEP_LAST, depth=100)


class StaticTfListener:
    """Feeds /tf_static into a tf2_ros.Buffer; the drop-in for TransformListener when only static frames are looked up."""

    def __init__(self, buffer, node):
        from tf2_msgs.msg import TFMessage
        self.buffer = buffer
        self.sub = node.create_subscription(TFMessage, '/tf_static', self._on_static, TF_STATIC_QOS)

    def _on_static(self, msg):
        for t in msg.transforms:
            self.buffer.set_transform_static(t, 'default_authority')


def tf_listener(buffer, node, static_only: bool):
    """StaticTfListener if static_only, else the full tf2_ros.TransformListener (/tf + /tf_static)."""
    if static_only:
        return StaticTfListener(buffer, node)
    from tf2_ros import TransformListener
    return TransformListener(buffer, node)
