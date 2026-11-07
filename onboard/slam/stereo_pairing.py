"""Pairing the two eyes of a stereo camera by capture stamp, with BOUNDED memory. PURE.

Used by the cuVSLAM tracker node (copied into its image next to it) and by the simulator's
stereo-depth node (generator/stereo_depth_node.py). Review 2026-10-17: both kept every frame of
an eye whose partner never came - 1000 unmatched frames in a probe, ~1 MB each in the depth node -
because unmatched halves were only discarded when a later pair completed.
"""
from __future__ import annotations

MAX_PENDING = 8                  # frames waiting for their partner (eyes normally arrive ms apart)
MAX_AGE_NS = 500_000_000         # a half older than the newest capture by this never pairs


def add_half(half, t_ns, side, item, max_pending=MAX_PENDING, max_age_ns=MAX_AGE_NS):
    """Add one eye's frame (side "l" or "r") captured at t_ns to `half` ({stamp: {side: item}}).
    -> (pair, dropped): pair = {"l": ..., "r": ...} when both eyes of t_ns are present (removed from
    `half`), else None; dropped = how many unmatched halves were discarded by this call."""
    t_ns = int(t_ns)
    h = half.setdefault(t_ns, {})
    h[side] = item
    dropped = 0
    pair = None
    if "l" in h and "r" in h:
        pair = half.pop(t_ns)
        for k in [k for k in half if k < t_ns]:          # older halves will never pair
            del half[k]
            dropped += 1
    if half:
        newest = max(max(half), t_ns)
        for k in [k for k in half if k < newest - max_age_ns]:
            del half[k]
            dropped += 1
        while len(half) > max_pending:
            del half[min(half)]
            dropped += 1
    return pair, dropped
