"""Validated PX4 policy inputs, independent of ROS. NED velocity and FRD->NED attitude."""
import math

import numpy as np


class Px4Feedback:
    def __init__(self, timeout_s=0.5, rf_max=4.0):
        self.timeout_s = timeout_s
        self.vel = self.q = None
        self.range_m = float(rf_max)  # until the first valid range; later gaps retain the last valid value
        self.range_t = None
        self.lpos_t = self.att_t = None
        self.fault = None
        self._samples = {}

    def _sample(self, key, stamp, resets, now, active):
        previous = self._samples.get(key)
        if stamp <= 0:
            if active:
                self.fault = f'{key}: invalid sample timestamp'
            return False
        if previous is not None:
            old_stamp, old_resets, old_time = previous
            if active and (resets != old_resets or stamp < old_stamp or
                           (stamp - old_stamp) * 1e-6 > self.timeout_s or
                           now - old_time > self.timeout_s):
                self.fault = f'{key}: estimate reset or feedback gap'
            if stamp == old_stamp:
                return False  # repeated samples cannot renew freshness
        self._samples[key] = stamp, resets, now
        return True

    def on_lpos(self, m, now, active=False):
        velocity = (float(m.vx), float(m.vy), float(m.vz))
        valid = (m.v_xy_valid and m.v_z_valid and m.heading_good_for_control and
                 all(math.isfinite(v) for v in velocity))
        range_ok = m.dist_bottom_valid and math.isfinite(m.dist_bottom) and m.dist_bottom > 0
        if not valid:
            self.vel = None
        if active and not valid:
            self.fault = 'PX4 velocity/heading invalid'
        if not self._sample('local position', m.timestamp_sample,
                            (m.vxy_reset_counter, m.vz_reset_counter, m.heading_reset_counter), now, active):
            return
        self.lpos_t = now
        self.vel = np.asarray(velocity) if valid else None
        if range_ok:
            self.range_m, self.range_t = float(m.dist_bottom), now

    def on_att(self, m, now, active=False):
        q = np.asarray(m.q, dtype=float)
        norm = float(np.linalg.norm(q))
        valid = q.shape == (4,) and np.isfinite(q).all() and abs(norm - 1.) < .01
        if not valid:
            self.q = None
            if active:
                self.fault = 'PX4 attitude invalid'
        if not self._sample('attitude', m.timestamp_sample, (m.quat_reset_counter,), now, active):
            return
        self.att_t = now
        self.q = q / norm if valid else None

    def error(self, now, timeout_s):
        if self.fault:
            return self.fault
        for name, stamp in (('PX4 velocity/range', self.lpos_t), ('PX4 attitude', self.att_t)):
            if stamp is None or not 0 <= now - stamp <= timeout_s:
                return name + ' missing or stale'
        if self.vel is None or self.q is None or self.range_m is None:
            return 'PX4 velocity, attitude or downward distance invalid'
        return None
